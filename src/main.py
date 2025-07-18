import datetime
from fileinput import filename
from genericpath import exists
import io
import subprocess
#import uuid

from flask import Flask, abort, render_template, request, jsonify, json, redirect, url_for, session, send_file, Response, send_from_directory
import pyodbc
import os
import cv2 as cv
import glob
import jwt
import base64
from datetime import datetime, timedelta
from shutil import copyfile
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import psycopg2
#from requests import patch
from werkzeug.utils import secure_filename

from Resources.QueriesProcedures import (validate_login_query,
                                         create_account_query,
                                         update_session,
                                         insert_new_frame,
                                         validate_frame_exists,
                                         update_frame_value,
                                         get_frames_query,
                                         get_users_query,
                                         check_email_query,
                                         get_menu_options_query,
                                         get_all_paths_query,
                                         get_tutorial_state_query,
                                         update_tutorial_state_query,
                                         insert_tutorial_path_query,
                                         validate_has_path_query,
                                         user_one_query,
                                         check_cedula_query,
                                         delete_user_query,
                                         edit_user_query,
                                         save_main_route_query,
                                         get_id_main_path_query,
                                         get_id_main_path_query_query,
                                         delete_folder_query
                                         )
from Resources.Middleware import token_required
from Resources.Middleware import get_key  # , deserialize_token
from model.PoseModule import poseDetector
from Resources.Conexion import get_connection
from Resources.Encrypt import encrypt_password
from model.BehaviorDetector import BehaviorDetector
from model.BehaviorDetector3d import BehaviorDetector3D
from Resources.Helper import get_work_path, get_processed_route, normalizeUrl

# Inicializar la app Flask
app = Flask(__name__, static_folder="static")
conn = get_connection()

# Crear una carpeta para guardar las imágenes subidas
UPLOAD_FOLDER = 'static/uploads'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = 'static/videos/uploads'
app.config['RESULT_FOLDER'] = 'static/videos/results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Inicializar el modulo de detección de poses
detector = poseDetector()
detecciones_guardadas = []

# region RENDER_VIEWS_ROUTES
# ROUTES
@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/upload-image')
def upload():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/create_account')
def create_accountView():
    return render_template('create-account.html')


@app.route('/configuration_path')
def configuration_path():
    return render_template('parametrizacion-rutas.html')


@app.route('/verify-images')
def verify_images():
    return render_template('verify-images.html')


@app.route('/dashboard')
def view_dashboard():
    print('view_dashboard')
    return render_template('dashboard.html')


@app.route('/gestion-usuarios')
def manage_users():
    return render_template('gestion-usuarios.html')


@app.route('/upload-videos')
def upload_videos():
    return render_template('upload-video.html')


@app.route('/live-detection')
def live_detection():
    return render_template('live-detection.html')

@app.route('/video-detection')
def video_detection():
    return render_template('video-detection.html')

# endregion

# region POSTURE_ESTIMATION_MODULE_ENDPOINTS
# ENDPOINTS
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Verificar si se recibió el nombre del archivo
        data = request.get_json()

        if not data or 'fileName' not in data:
            return jsonify({'message': 'No se proporcionó un nombre de archivo válido.'}), 400

        filename = data['fileName']
        filepath = os.path.join('src', UPLOAD_FOLDER, filename)

        # Verificar si el archivo existe
        if not os.path.isfile(filepath):
            return jsonify({'message': 'El archivo no existe.'}), 404

        with Image.open(filepath) as image:
            original_width, original_height = image.size
            if original_width > original_height:
                image_position = 'horizontal'
            elif original_width < original_height:
                image_position = 'vertical'
            else:
                image_position = 'cuadrada'

        # Procesar la imagen y detectar la pose
        img = cv.imread(filepath)

        img_with_pose = detector.findPose(img)

        # Guardar la imagen con los puntos detectados
        output_path = os.path.join("src", UPLOAD_FOLDER, 'points_' + filename)
        cv.imwrite(output_path, img_with_pose)

        # Coordenadas de los puntos
        position = detector.findPosition(img_with_pose)

        return jsonify({
            'message': 'Imagen procesada exitosamente.',
            'path': output_path,
            'image_pos': image_position,
            'position': position
        }), 200

    except Exception as e:
        return jsonify({'message': f'Error interno del servidor: {str(e)}'}), 500


@app.route('/resize_image', methods=['POST'])
def resizeImage():
    try:
        width = int(request.form.get('width'))
        height = int(request.form.get('height'))

        # Verificar si el archivo está en la solicitud
        if 'image' not in request.files:
            return jsonify({'error': 'No se encontró ninguna imagen'}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400

        # Guardar el archivo temporalmente
        filepath = os.path.join("src", UPLOAD_FOLDER, image.filename)
        image.save(filepath)

        # Obtener dimensiones de la imagen antes de redimensionar
        with Image.open(filepath) as img:
            # original_width, original_height = img.size  # Obtiene ancho y alto
            original_format = img.format

            if width > height:
                with Image.open(filepath) as img:
                    resized_img = img.resize((width, height))
                    resized_img_w = width
                    resized_img_h = height
                    resized_image_name = 'resized_' + image.filename
                    output_path_file = os.path.join('src', UPLOAD_FOLDER, resized_image_name)

                    try:
                        if original_format == 'PNG':
                            resized_img.save(output_path_file, format='PNG')
                        else:
                            resized_img.save(output_path_file, format='JPEG', quality=90)
                    except Exception as e:
                        print(f"Error al guardar la imagen: {e}")

            else:
                # Si la Imagen es vertical
                with Image.open(filepath) as img:
                    resized_img = img.resize((width, height))
                    resized_img_w = width
                    resized_img_h = height
                    resized_image_name = 'resized_' + image.filename
                    output_path_file = os.path.join(UPLOAD_FOLDER, resized_image_name)

                    # Guardar la imagen redimensionada
                    resized_img.save(output_path_file, format='JPEG', quality=90)

        return jsonify({'Imagen_Redimensionada': output_path_file, 'alto': resized_img_h, 'ancho': resized_img_w}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/resize_image_params', methods=['POST'])
def resize_image_params():
    try:
        width = request.form.get('width')
        height = request.form.get('height')
        image = request.form.get('image')

        # Guardar el archivo temporalmente
        filepath = os.path.join(UPLOAD_FOLDER, image)

        # Obtener dimensiones de la imagen antes de redimensionar
        with Image.open(filepath) as img:
            original_width, original_height = img.size  # Obtiene ancho y alto
            if original_width > original_height:
                with Image.open(filepath) as img:
                    resized_img = img.resize((int(width), int(height)))
                    resized_img_w = width
                    resized_img_h = height
                    resized_image_name = 'new_' + image
                    output_path_file = os.path.join(UPLOAD_FOLDER, resized_image_name)

                    # Guardar la imagen redimensionada
                    resized_img.save(output_path_file, format='JPEG', quality=90)
                    img = cv.imread(output_path_file)
                    img_with_pose = detector.findPose(img)

                    # Coordenadas de los puntos
                    position = detector.findPosition(img_with_pose)

                    return jsonify({
                        'message': 'Imagen procesada exitosamente.',
                        'path': output_path_file,
                        'position': position
                    }), 200

            else:
                # Si la Imagen es vertical
                with Image.open(filepath) as img:
                    resized_img = img.resize((int(width), int(height)))
                    resized_img_w = width
                    resized_img_h = height
                    resized_image_name = 'new_' + image
                    output_path_file = os.path.join(UPLOAD_FOLDER, resized_image_name)

                    # Guardar la imagen redimensionada
                    resized_img.save(output_path_file, format='JPEG', quality=90)
                    img = cv.imread(output_path_file)
                    img_with_pose = detector.findPose(img)

                    # Coordenadas de los puntos
                    position = detector.findPosition(img_with_pose)

                    return jsonify({
                        'message': 'Imagen procesada exitosamente.',
                        'path': output_path_file,
                        'position': position
                    }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save', methods=['POST'])
def saveImageData():
    try:
        data = request.get_json()
        file_name = data['data']['file']
        points_position = data['data']['points_position']
        width_file = data['data']['width']
        heigth_file = data['data']['height']
        file = data['data']['pathToSave']
        center_x = width_file / 2
        center_y = heigth_file / 2
        local_path = str(file)
        final_path = local_path + '\\Imagen'
        print('ruta final:', final_path)

        # si la ruta Imagen no existe la crea
        if not os.path.exists(final_path):
            os.makedirs(final_path)

        # obtener archivos totales
        total = get_total_files(final_path)

        # nuevo formato de nombre  000001
        name, ext = os.path.splitext(file_name)
        name_to_save = f"{total}{ext}"

        # Guardar el JSON con los puntos
        json_file_name = "Points_Json.json"
        path_json = os.path.join(local_path, json_file_name)
        print('path_json:', path_json)

        new_image_json = {
            "path": name_to_save,
            "content": {
                "points_position": points_position
            },
            "size": {
                "width": width_file,
                "heigth": heigth_file
            },
            "center": {
                "x": center_x,
                "y": center_y
            }
        }

        if os.path.exists(path_json):
            with open(path_json, "r") as json_file:
                data = json.load(json_file)
                if not isinstance(data, list):
                    data = [data]
        else:
            data = []
        data.append(new_image_json)

        with open(path_json, "w") as json_file:
            json.dump(data, json_file, indent=4)

        image_file_path = os.path.join(final_path, name_to_save)
        path_base_image = os.path.join(UPLOAD_FOLDER, f"points_{file_name}")

        # Crear o copiar la imagen con el nombre indicado
        copyfile(path_base_image, image_file_path)  # Copia una imagen base con el nuevo nombre

        delete_temp_image(UPLOAD_FOLDER)
        return jsonify({'success': True, 'message': 'Imagen y datos guardados exitosamente!'}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
    return jsonify({'success': False, 'message': 'Ocurrió un error al guardar la imagen o los datos.'}), 500


def delete_temp_image(carpeta):
    files = glob.glob(os.path.join(carpeta, "*"))  # Lista todos los archivos
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def get_total_files(final_path):
    file_count = sum(1 for file in os.listdir(final_path)
                     if os.path.isfile(os.path.join(final_path, file)))
    file_count_str = str(file_count + 1)

    if len(file_count_str) == 1:
        final_path = '0000' + file_count_str

    elif len(file_count_str) == 2:
        final_path = '000' + file_count_str
    elif len(file_count_str) == 3:
        final_path = '00' + file_count_str
    elif len(file_count_str) == 4:
        final_path = '0' + file_count_str
    else:
        final_path = file_count_str

    return final_path


# user controllers
@app.route('/validateLogin', methods=['POST'])
def validate_login():
    try:
        mail = request.form.get('mail')
        passw = encrypt_password(request.form.get('pass'))
        with get_connection() as conn:
            cursor = conn.cursor()
            query = validate_login_query()
            cursor.execute(query, (mail,))
            result = cursor.fetchone()

            if result and result[3] == passw and result[5] == '1':
                token = jwt.encode({
                    "user_id": result[0],
                    "exp": datetime.utcnow() + timedelta(hours=8)
                }, get_key(), algorithm="HS256")

                update_session_login(result)

                return jsonify({'authenticated': True, 'redirect_url': url_for('view_dashboard'), 'user': result[1],
                                'id': result[0], 'idRol': result[6], 'stateuser': result[5], 'token': token}), 200
            else:
                return jsonify(
                    {'authenticated': False, 'message': 'Correo o contraseña incorrecta', 'stateuser': result[5]}), 401

    except psycopg2.Error as e:
        print("Error al ejecutar la consulta:", e)
        return jsonify({'authenticated': False, 'message': 'Error interno del servidor'}), 500


def update_session_login(result):
    with get_connection() as conn:
        fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = update_session()
        with conn.cursor() as cursor:
            cursor.execute(query, (fecha_actual, result[0]))
            conn.commit()


@app.route('/createAccount', methods=['POST'])
def createAccount():
    try:
        name = request.form.get('name')
        lastName = request.form.get('lastName')
        identification = request.form.get('identification')
        email = request.form.get('email')
        password = request.form.get('pass')
        passw = encrypt_password(password)

        with get_connection() as conn:
            cursor = conn.cursor()
            query = create_account_query()
            params = (name, lastName, identification, passw, email)
            cursor.execute(query, params)
            conn.commit()

            return jsonify({'created': True, 'redirect_url': url_for('login')}), 200


    except psycopg2.Error as e:
        return jsonify({'authenticated': False, 'message': 'Error interno del servidor'}), 500


@app.route('/get_menu_options', methods=['GET'])
def get_menu_option():
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            query = get_menu_options_query()
            cursor.execute(query)
            result = cursor.fetchone()
        if result is not None and len(result) > 0:
            return jsonify({'options': result[0]}), 200
        else:
            return jsonify({'error': 'No se encontraron resultados'}), 404

    except psycopg2.Error as e:
        return jsonify({'authenticated': False, 'message': 'Error interno del servidor'}), 500


@app.route('/users-all', methods=['GET'])
def getUsers():
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            query = get_users_query()
            cursor.execute(query)
            rows = cursor.fetchall()

            usuarios = []
            for row in rows:
                usuarios.append({
                    "id": row[0],
                    "nombre": row[1],
                    "apellido": row[2],
                    "cedula": row[3],
                    "mail": row[4],
                    "nombrerol": row[5],
                    "stateUser": row[6],
                })

            return jsonify(usuarios), 200

    except pyodbc.Error as e:
        return jsonify({'authenticated': False, 'message': 'Error interno del servidor'}), 500


@app.route('/user-one', methods=['POST'])
def user_one():
    try:
        data = request.get_json()
        id = data.get('id')

        if not id:
            return jsonify({'result': None, 'message': 'ID no proporcionado'}), 400

        with get_connection() as conn:
            cursor = conn.cursor()
            query = user_one_query()
            cursor.execute(query, (id,))
            result = cursor.fetchone()

            if result:
                return jsonify({'result': result}), 200
            else:
                return jsonify({'result': None, 'message': 'Usuario no encontrado'}), 404

    except Exception as e:
        return jsonify({'error': 'Error interno del servidor', 'details': str(e)}), 500


@app.route('/check_cedula', methods=['POST'])
def check_cedula():
    try:
        data = request.get_json()
        cedula = data.get('cedula')

        with get_connection() as conn:
            cursor = conn.cursor()
            query = check_cedula_query()
            cursor.execute(query, (cedula,))
            exits = cursor.fetchone()[0] > 0

            return jsonify({'exists': exits}), 200
    except Exception as e:
        return jsonify({'error': 'Error interno del servidor'}), 500


@app.route('/check_email', methods=['POST'])
def check_email():
    try:
        data = request.get_json()
        email = data.get('email')

        with get_connection() as conn:
            cursor = conn.cursor()
            query = check_email_query()
            cursor.execute(query, (email,))
            exits = cursor.fetchone()[0] > 0

            return jsonify({'exists': exits}), 200
    except Exception as e:
        return jsonify({'error': 'Error interno del servidor'}), 500


@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_to_delete = request.form.get('user')
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            query = delete_user_query()
            params = (user_to_delete,)
            result = cursor.execute(query, params)
            conn.commit()
            return jsonify({
                'result': True,
                'message': 'Usuario Eliminado'
            }), 200

    except psycopg2.errors.ForeignKeyViolation:
        return jsonify({
            'result': False,
            'message': 'No se puede eliminar el usuario porque tiene rutas activas.'
        }), 400

    except pyodbc.Error as e:
        conn.rollback()
        return jsonify({'result': False, 'message': e}), 500


@app.route('/edit_user', methods=['POST'])
def edit_user():
    name = request.form.get('name')
    lastName = request.form.get('lastName')
    identification = request.form.get('identification')
    user_id = request.form.get('user_id')
    user_rol = request.form.get('user_rol')
    user_state = request.form.get('user_state')

    if not user_id:
        return jsonify({'result': False, 'message': 'ID de usuario no proporcionado'}), 400

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            query = edit_user_query()
            params = (name, lastName, identification, user_rol, user_state, user_id)
            cursor.execute(query, params)
            conn.commit()

            if cursor.rowcount > 0:
                return jsonify({'result': True, 'message': 'Usuario actualizado'}), 200
            else:
                return jsonify({'result': False, 'message': 'Usuario no encontrado'}), 404
    except Exception as e:
        conn.rollback()
        return jsonify({'result': False, 'message': 'Error interno del servidor', 'error': str(e)}), 500


@app.route('/parametrizador-ruta-principal', methods=['POST'])
def save_main_route():
    user_id = request.form.get('id')
    main_path = request.form.get('path')

    with get_connection() as conn:
        cursor = conn.cursor()
        query = save_main_route_query()
        params = (main_path, user_id)
        cursor.execute(query, params)
        conn.commit()

        return jsonify({'created': True, 'message': 'Ruta guardada'}), 200


@app.route('/validate_has_path', methods=['POST'])
def HasPath():
    user_id = int(request.form.get('id'))
    with (get_connection() as conn):
        cursor = conn.cursor()
        query = validate_has_path_query()
        params = (user_id,)
        cursor.execute(query, params)
        row = cursor.fetchone()

        if row:
            return jsonify({'ruta': row[0], 'message': 'Ruta guardada'}), 200
    return jsonify({'message': 'Ocurrio un error al tratar de guardar la ruta', 'total': 0}), 400


@app.route('/getIdMainPath', methods=['POST'])
def get_id_main_path():
    path = request.form.get('main_path')
    with get_connection() as conn:
        cursor = conn.cursor()
        query = get_id_main_path_query()
        params = (path,)
        cursor.execute(query, params)
        row = cursor.fetchone()

        if row:
            return jsonify({'id_path': row[0]}), 200
    return jsonify({'Ocurrio un error al obtener el id de la ruta'}), 400


@app.route('/save_new_folder', methods=['POST'])
def save_new_folder():
    id_main_path = int(request.form.get('id_main_folder'))
    main_path = request.form.get('nameFolder')
    #date = datetime.now().strftime('%d-%m-%Y')
    date = datetime.now().strftime('%Y-%m-%d')

    # Se crea la carpeta automaticamente
    if not os.path.exists(main_path):
        os.makedirs(main_path)

        with get_connection() as conn:
            cursor = conn.cursor()
            query = get_id_main_path_query_query()
            params = (main_path,date,id_main_path)
            cursor.execute(query, params)
            conn.commit()

            return jsonify({'created': True, 'message': 'Ruta creada con exito'}), 200
    else:
        return jsonify({'created': False, 'message': 'Ruta ya existe'}), 400


@app.route('/all_paths', methods=['POST'])
def getPaths():
    id = int(request.form.get('id'))
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            query = get_all_paths_query()
            params = (id,)
            cursor.execute(query, params)
            rows = cursor.fetchall()

            rutas = []
            for row in rows:
                rutas.append({
                    "id": row[0],
                    "nombre": row[1],
                    "fechaCreacion": row[2]
                })
            return jsonify(rutas), 200

    except pyodbc.Error as e:
        return jsonify({'authenticated': False, 'message': 'Error interno del servidor'}), 500


@app.route('/delete_folder', methods=['POST'])
def delete_folder():
    path_to_delete = request.form.get('path')
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            query = delete_folder_query()
            params = (path_to_delete,)
            cursor.execute(query, params)
            conn.commit()
            os.rmdir(path_to_delete)

            return jsonify({
                'result': True,
                'message': 'Ruta Eliminada'
            })

    except pyodbc.Error as e:
        conn.rollback()
        return jsonify({'result': False, 'message': 'Error interno del servidor'}), 500


@app.route('/getFilesByPathname', methods=['POST'])
def get_files_by_pathname():
    pathName = request.form.get('pathName')
    files_path = os.path.join(pathName, "Imagen")

    try:
        files = os.listdir(files_path)
    except FileNotFoundError:
        return jsonify({"error": "La carpeta 'files' no se encuentra en la ruta proporcionada."}), 404

    json_file = next((file for file in os.listdir(pathName) if file.lower().endswith('.json')), None)

    if not json_file:
        return jsonify({"error": "No se encontró un archivo JSON en la ruta proporcionada."}), 404

    json_file_path = os.path.join(pathName, json_file)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_files = [{"nombre": file} for file in files]

        return jsonify({'data': data, 'files': total_files})

    except json.JSONDecodeError:
        return jsonify({"error": "Error al leer el archivo JSON."}), 400
    except Exception as e:
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500


@app.route('/getFiles', methods=['POST'])
def get_files():
    pathName = request.form.get('pathName')
    files = os.listdir(pathName)
    json_file = next((file for file in files if file.lower().endswith('.json')), None)

    file_path = os.path.join(pathName, json_file)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError:
        return jsonify({"error": "Error al leer el archivo JSON."}), 400


@app.route('/getImage', methods=['POST'])
def get_image():
    pathName = request.form.get('path')
    file = request.form.get('file')
    pathImage = os.path.join(pathName, 'Imagen')
    all_path = os.path.join(pathImage, file)
    extension = os.path.splitext(file)[-1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
    }

    mimetype = mime_types.get(extension)

    return send_file(all_path, mimetype=mimetype)


# CONTROLLER FOR VIDEOS VIEW

@app.route('/generate_images_from_videos', methods=['POST'])
def generate_images_from_videos():
    fps_value = int(request.form.get('fps_value'))

    if 'video' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400

    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)

    try:
        # Guardar el archivo de video temporalmente
        video_file.save(video_path)
    except Exception as e:
        return jsonify({'error': f'Error al guardar el archivo: {str(e)}'}), 500

    try:
        frames = []
        with VideoFileClip(video_path) as clip:
            max_duration = 30
            fps = fps_value
            frame_times = [i / fps for i in range(int(min(clip.duration, max_duration) * fps))]

            for idx, time in enumerate(frame_times):
                frame = clip.get_frame(time)
                img = io.BytesIO()
                Image.fromarray(frame).save(img, format='JPEG')
                img.seek(0)
                encoded_frame = base64.b64encode(img.getvalue()).decode('utf-8')
                frames.append((f"frame_{idx}.jpg", encoded_frame))
        os.remove(video_path)

        return jsonify({frame_name: frame_data for frame_name, frame_data in frames}), 200

    except Exception as e:
        return jsonify({'error': f'Error procesando el video: {str(e)}'}), 500


@app.route('/upload_image_video', methods=['POST'])
def upload_image_from_video():
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'No se envió ninguna imagen.'}), 400

        image_file = request.files['image']
        filename = image_file.filename

        if filename == '':
            return jsonify({'message': 'El nombre del archivo está vacío.'}), 400

        try:
            width = int(request.form.get('width', 0))
            height = int(request.form.get('height', 0))
        except ValueError:
            return jsonify({'message': 'Los valores de width y height deben ser números enteros.'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        try:
            with Image.open(filepath) as image:
                original_width, original_height = image.size

                if original_width > original_height:
                    final_width, final_height = width, height
                elif original_height > original_width:
                    final_width, final_height = width, height
                else:
                    final_width = final_height = min(width, height) if width and height else 300  # Valor por defecto

                # resized_img = image.resize((300, 445))
                resized_img = image.resize((final_width, final_height))
                resized_image_name = 'resized_' + filename
                output_path_file = os.path.join(UPLOAD_FOLDER, resized_image_name)
                resized_img.save(output_path_file)
        except Exception as e:
            return jsonify({'message': f'Error al procesar la imagen: {str(e)}'}), 500

        img = cv.imread(output_path_file)
        if img is None:
            return jsonify({'message': 'Error al cargar la imagen con OpenCV.'}), 400

        try:
            img_with_pose = detector.findPose(img)
            position = detector.findPosition(img_with_pose)
        except Exception as e:
            return jsonify({'message': f'Error al detectar la pose: {str(e)}'}), 500

        output_path = os.path.join(UPLOAD_FOLDER, 'points_' + filename)
        cv.imwrite(output_path, img_with_pose)

        if original_width > original_height:
            image_position = 'horizontal'
        elif original_height > original_width:
            image_position = 'vertical'
        else:
            image_position = 'cuadrada'

        return jsonify({
            'message': 'Imagen procesada exitosamente.',
            'path': output_path,
            'image_pos': image_position,
            'position': position,
            'filename': 'points_' + filename,
            'height': final_height,
            'width': final_width
        }), 200

    except Exception as e:
        return jsonify({'message': f'Error interno del servidor: {str(e)}'}), 500


@app.route('/save_image_from_video', methods=['POST'])
def save_image_from_video():
    try:
        data = request.get_json()
        file_name = data['data']['file']
        points_position = data['data']['points_position']
        width_file = data['data']['width']
        heigth_file = data['data']['height']
        file = data['data']['pathToSave']
        center_x = width_file / 2
        center_y = heigth_file / 2
        local_path = str(file)
        final_path = local_path + '\\Imagen'

        # si la ruta Imagen no existe la crea
        if not os.path.exists(final_path):
            os.makedirs(final_path)

        # obtener archivos totales
        total = get_total_files(final_path)

        # nuevo formato de nombre  000001
        name, ext = os.path.splitext(file_name)
        name_to_save = f"{total}{ext}"

        # Guardar el JSON con los puntos
        json_file_name = "Points_Json.json"
        path_json = os.path.join(local_path, json_file_name)

        new_image_json = {
            "path": name_to_save,
            "content": {
                "points_position": points_position
            },
            "size": {
                "width": width_file,
                "heigth": heigth_file
            },
            "center": {
                "x": center_x,
                "y": center_y
            }
        }

        if os.path.exists(path_json):
            with open(path_json, "r") as json_file:
                data = json.load(json_file)
                if not isinstance(data, list):
                    data = [data]
        else:
            data = []
        data.append(new_image_json)

        with open(path_json, "w") as json_file:
            json.dump(data, json_file, indent=4)

        image_file_path = os.path.join(final_path, name_to_save)
        path_base_image = os.path.join(UPLOAD_FOLDER, file_name)

        # Crear o copiar la imagen con el nombre indicado
        copyfile(path_base_image, image_file_path)  # Copia una imagen base con el nuevo nombre

        delete_temp_image(UPLOAD_FOLDER)
        return jsonify({'success': True, 'message': 'Imagen y datos guardados exitosamente!'}), 200

    except Exception as e:
        return jsonify({'success': False, 'message': 'Ocurrió un error al guardar la imagen o los datos.'}), 500


def validate_frame_exists(id_user):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            query = "SELECT COUNT(*) FROM parametrizador_fps WHERE id_user = %s"
            params = (id_user,)
            cursor.execute(query, params)
            count = cursor.fetchone()[0]  # Obtener el resultado de la consulta
            print(count)  # Imprimir el resultado
            return count


@app.route('/saveNewFrame', methods=['POST'])
def save_frames_for_video():
    id_user = request.form.get('id_user')
    frame_value = request.form.get('frame_value')

    exist = validate_frame_exists(id_user)
    print(exist)
    if exist > 0:
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    query = update_frame_value()
                    params = (frame_value, id_user)
                    cursor.execute(query, params)

                    return jsonify({
                        'result': True,
                        'message': 'Parametrización de FPS actualizado'
                    })
        except pyodbc.Error as e:
            return jsonify({'result': False, 'message': 'Error interno del servidor'}), 500
    else:
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    query = insert_new_frame()
                    params = (frame_value, id_user)
                    cursor.execute(query, params)
            return jsonify({
                'result': True,
                'message': 'Parametrizacion de FPS registrada'
            })
        except pyodbc.Error as e:
            return jsonify({'result': False, 'message': 'Error interno del servidor'}), 500


@app.route('/get_frames', methods=['POST'])
def get_frames():
    id_user = request.form.get('id')
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = get_frames_query()
                params = (id_user,)
                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    return jsonify({'result': True, 'response': result[0]})
                else:
                    return jsonify({'result': False, 'message': 'No se encontraron datos'})

    except pyodbc.Error as e:
        return jsonify({'result': False, 'message': 'Error interno del servidor'}), 500


@app.route('/getTutorialState', methods=['POST'])
@token_required
def get_tutorial_state():
    id_user = int(request.form.get('id'))

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = get_tutorial_state_query()
                params = (id_user,)
                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    return jsonify({'result': True, 'state_tutorial': result[0]})
                else:
                    return jsonify({'result': False, 'message': 'No se encontraron datos'})

    except pyodbc.Error as e:
        return jsonify({'result': False, 'message': 'Error interno del servidor'}), 500


@app.route('/saveFirstTutorialInfo', methods=['POST'])
def save_first_tutorial_info():
    id_user = request.form.get('id_user')
    fps_value = request.form.get('fps_value')
    path = request.form.get('main_path')

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = insert_new_frame()
                params = (fps_value, id_user)
                cursor.execute(query, params)

                query = insert_tutorial_path_query()
                params = (path, id_user)
                cursor.execute(query, params)

                query = update_tutorial_state_query()
                params = (id_user,)
                cursor.execute(query, params)

            return jsonify({'result': True, 'message': 'Datos registrados correctamente'}), 200

    except Exception as e:
        print(f"Error al ejecutar las consultas: {e}")
        return jsonify({'result': False, 'message': 'Error interno del servidor'}), 500


# endregion

# region SUSPICIOUS_DETECTION_MODULE_ENDPOINTS

@app.route('/save-video', methods=['POST'])
def save_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se encontró el archivo de video'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    file_path = get_work_path()
    full_path = os.path.join(file_path, file.filename)

    if not os.path.exists(full_path):
        file.save(full_path)
        return jsonify({'message': 'Archivo guardado', 'filename': file.filename}), 200
    else:
        return jsonify({'message': 'El archivo ya existe', 'filename': file.filename}), 400


@app.route('/stream_frames/<filename>/<frame_skip>/<dimension>')
def stream_frames(filename, frame_skip, dimension):
    file_path = get_work_path()
    filepath = os.path.join(file_path, filename)

    if os.path.exists(os.path.join(file_path, 'Processeds', 'Processed' + filename)):
        os.remove(os.path.join(file_path,'Processeds', 'Processed' + filename))

    # Procesar el video
    output_path = os.path.join(file_path, 'Processeds', 'Processed' + os.path.splitext(filename)[0] + '.webm')

    def generar():
        global detecciones_guardadas
        detector = BehaviorDetector(frame_skip=int(frame_skip)) if dimension == '2D' else BehaviorDetector3D(frame_skip=int(frame_skip))
        
        for result in detector.process_video(filepath, output_path):
            if isinstance(result, list):  # Si es la lista de detecciones
                detecciones_guardadas = result
                yield "data: EOF\n\n"
            else:  # Si es un frame en base64 (string) o un diccionario con frame+detecciones
                if isinstance(result, dict):
                    import json
                    json_data = json.dumps(result)
                    yield f"data: {json_data}\n\n"
                else:
                    # Si es solo un string (frame en base64)
                    yield f"data: {result}\n\n"

    return Response(generar(), mimetype='text/event-stream')


@app.route('/camera_stream_frames/<frame_skip>/<dimension>/<connection>')
def camera_stream_frames(frame_skip, dimension, connection):
    def generar():
        global detecciones_guardadas
        detector = BehaviorDetector(frame_skip=int(frame_skip), connection=normalizeUrl(connection), with_camera=True) if dimension == '2D' else BehaviorDetector3D(frame_skip=int(frame_skip), connection=normalizeUrl(connection), with_camera=True)
        
        for result in detector.process_video(None, None):
            if isinstance(result, list):  # Si es la lista de detecciones
                detecciones_guardadas = result
                yield "data: EOF\n\n"
            else:  # Si es un frame en base64 (string) o un diccionario con frame+detecciones
                if isinstance(result, dict):
                    import json
                    json_data = json.dumps(result)
                    yield f"data: {json_data}\n\n"
                else:
                    # Si es solo un string (frame en base64)
                    yield f"data: {result}\n\n"

    return Response(generar(), mimetype='text/event-stream')

@app.route('/detecciones')
def get_detecciones():
    return jsonify(detecciones_guardadas)

@app.route('/processed-video/<filename>')
def get_video(filename):
    video_path = os.path.join(get_processed_route(), "Processed" + filename)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/webm')
    else:
        abort(404, description="Video no encontrado")
        
# endregion

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
