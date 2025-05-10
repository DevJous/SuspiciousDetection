from multiprocessing import process
import cv2
import os
import shutil
import mediapipe as mp
import numpy as np
import time
import math
from collections import defaultdict, deque
from Resources.Helper import get_temp_route, format_number
import base64


class BehaviorDetector:
    def __init__(self, frame_skip=3):
        # Inicializar MediaPipe Pose y Hands
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        # Nuevo parámetro para configurar cuántos frames saltar
        self.frame_skip = frame_skip

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Agregar FaceMesh para detección facial y de mirada más precisa
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Variables para seguimiento de comportamientos (una sola persona)
        self.person_data = {
            'positions': deque(maxlen=30),  # Posiciones recientes (últimos 30 frames)
            'hidden_hands_frames': 0,  # Contador de frames con manos ocultas
            'hidden_hands_duration': 0,  # Duración de manos ocultas (en segundos)
            'hidden_hands_position': None,  # Posición donde se detectaron manos ocultas
            'suspicious_start_times': {},  # Tiempo de inicio de comportamientos sospechosos
            'alerted': set(),  # Comportamientos para los que ya se ha alertado

            # Variables para mirada excesiva
            'gaze_directions': deque(maxlen=60),  # Historial de direcciones de mirada (2 segundos @ 30fps)
            'gaze_change_counter': 0,  # Contador de cambios significativos de mirada
            'last_significant_gaze_time': 0,  # Timestamp del último cambio significativo

            # Variables para movimientos erráticos
            'pose_keypoints_history': deque(maxlen=15),  # Historial de posiciones de keypoints
            'movement_stability': deque(maxlen=30),  # Medidas de estabilidad
            'jerk_values': deque(maxlen=10),  # Valores de cambio brusco (jerk)
            'tremor_detected_frames': 0  # Frames consecutivos con temblor detectado
        }

        # Umbrales para detección
        self.hidden_hands_frame_threshold = 25  # Frames consecutivos con manos ocultas
        self.hidden_hands_time_threshold = 3.0  # Tiempo mínimo con manos ocultas (segundos)

        # Umbrales para detección de mirada excesiva
        self.gaze_angle_threshold = 0.3  # Cambio mínimo en radianes (~17 grados) para considerar cambio de mirada
        self.gaze_changes_threshold = 8  # Número de cambios en ventana de tiempo para considerar mirada excesiva
        self.gaze_time_window = 5.0  # Ventana de tiempo para evaluar cambios de mirada (segundos)

        # Umbrales para detección de movimientos erráticos
        self.jerk_threshold = 0.02  # Umbral para considerar un movimiento como brusco
        self.tremor_amplitude_threshold = 0.005  # Umbral para considerar una oscilación como temblor
        self.tremor_frames_threshold = 10  # Frames consecutivos con tremor para alertar
        self.stability_threshold = 0.01  # Umbral para considerar postura inestable

        # Factor de confianza para reducir falsos positivos
        self.confidence_threshold = 0.7

        # Puntos de referencia importantes para el seguimiento facial
        # Índices aproximados en el modelo de 468 puntos de FaceMesh
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 374, 373, 390]
        self.IRIS_INDICES = [468, 469, 470, 471, 472, 473]  # Si están disponibles en el modelo usado

    def detect_hand_pockets(self, pose_landmarks, hand_landmarks, frame_shape):
        """Detecta si las manos están en los bolsillos o detrás de la espalda"""
        if not pose_landmarks:
            return False

        h, w, _ = frame_shape

        # Extraer coordenadas relevantes
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calcular punto medio de los hombros para referencia
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2

        # Verificar si hay detecciones de manos
        visible_hands = set()
        if hand_landmarks:
            for hand_lm in hand_landmarks:
                # Calcular punto medio de la mano
                hand_x = sum(lm.x for lm in hand_lm.landmark) / len(hand_lm.landmark)

                # Determinar si es mano izquierda o derecha basado en la posición relativa
                if hand_x < shoulder_mid_x:
                    visible_hands.add('left')
                else:
                    visible_hands.add('right')

        # Verificar si las muñecas están cerca de las caderas (bolsillos) o detrás
        hands_in_pockets = False

        # Mano izquierda en bolsillo o detrás
        if 'left' not in visible_hands:
            hip_wrist_distance_left = math.sqrt((left_hip.x - left_wrist.x) ** 2 + (left_hip.y - left_wrist.y) ** 2)
            behind_back_left = left_wrist.z > left_hip.z + 0.1  # Z mayor significa más atrás
            if hip_wrist_distance_left < 0.15 or behind_back_left:
                hands_in_pockets = True

        # Mano derecha en bolsillo o detrás
        if 'right' not in visible_hands:
            hip_wrist_distance_right = math.sqrt(
                (right_hip.x - right_wrist.x) ** 2 + (right_hip.y - right_wrist.y) ** 2)
            behind_back_right = right_wrist.z > right_hip.z + 0.1  # Z mayor significa más atrás
            if hip_wrist_distance_right < 0.15 or behind_back_right:
                hands_in_pockets = True

        return hands_in_pockets

    def detect_excessive_gaze(self, face_landmarks, pose_landmarks, frame_shape, current_time):
        """Detecta si una persona mira excesivamente en diferentes direcciones o a la cámara"""
        if not face_landmarks and not pose_landmarks:
            return False

        # Si tenemos face_landmarks, usar para detección más precisa
        if face_landmarks:
            # Calcular dirección de la mirada usando la malla facial
            left_eye_center = np.mean([[face_landmarks.landmark[idx].x,
                                        face_landmarks.landmark[idx].y]
                                       for idx in self.LEFT_EYE_INDICES], axis=0)

            right_eye_center = np.mean([[face_landmarks.landmark[idx].x,
                                         face_landmarks.landmark[idx].y]
                                        for idx in self.RIGHT_EYE_INDICES], axis=0)

            # Punto entre los ojos como referencia
            eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)

            # Intentar usar iris si está disponible para mayor precisión
            has_iris_data = len(face_landmarks.landmark) > 468  # Verificar si el modelo tiene puntos de iris

            if has_iris_data:
                # Calcular centro del iris
                left_iris = np.mean([[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
                                     for idx in self.IRIS_INDICES[:3]], axis=0)
                right_iris = np.mean([[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
                                      for idx in self.IRIS_INDICES[3:]], axis=0)

                # Vector de dirección combinando ambos iris
                gaze_vector = np.mean([
                    left_iris - left_eye_center,
                    right_iris - right_eye_center
                ], axis=0)
            else:
                # Si no hay datos de iris, usar la posición de la nariz respecto al centro de los ojos
                nose_tip = np.array([face_landmarks.landmark[4].x, face_landmarks.landmark[4].y])
                gaze_vector = nose_tip - eyes_center

        # Si no hay face_landmarks pero tenemos pose_landmarks, usar estos puntos faciales básicos
        elif pose_landmarks:
            # Usar landmarks faciales del pose detector
            nose = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
                             pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y])

            left_eye = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x,
                                 pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y])

            right_eye = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x,
                                  pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y])

            eyes_center = np.mean([left_eye, right_eye], axis=0)
            gaze_vector = nose - eyes_center

        # Normalizar vector
        gaze_magnitude = np.linalg.norm(gaze_vector)
        if gaze_magnitude > 0:
            gaze_vector = gaze_vector / gaze_magnitude
        else:
            return False

        # Guardar dirección de mirada actual
        self.person_data['gaze_directions'].append(gaze_vector)

        # Necesitamos al menos 2 direcciones para comparar
        if len(self.person_data['gaze_directions']) < 2:
            return False

        # Comparar con dirección anterior para detectar cambio significativo
        prev_gaze = self.person_data['gaze_directions'][-2]
        dot_product = np.clip(np.dot(gaze_vector, prev_gaze), -1.0, 1.0)
        angle_change = np.arccos(dot_product)

        # Si el cambio es significativo, incrementar contador
        if angle_change > self.gaze_angle_threshold:
            self.person_data['gaze_change_counter'] += 1
            self.person_data['last_significant_gaze_time'] = current_time

            # Iniciar seguimiento del comportamiento si es el primer cambio detectado
            if 'excessive_gaze' not in self.person_data['suspicious_start_times']:
                self.person_data['suspicious_start_times']['excessive_gaze'] = current_time

        # Comprobar si se han producido suficientes cambios en la ventana de tiempo
        if 'excessive_gaze' in self.person_data['suspicious_start_times']:
            elapsed_time = current_time - self.person_data['suspicious_start_times']['excessive_gaze']

            # Restablecer contador si ha pasado demasiado tiempo desde el último cambio significativo
            if current_time - self.person_data['last_significant_gaze_time'] > self.gaze_time_window:
                self.person_data['gaze_change_counter'] = 0
                del self.person_data['suspicious_start_times']['excessive_gaze']
                if 'excessive_gaze' in self.person_data['alerted']:
                    self.person_data['alerted'].remove('excessive_gaze')
                return False

            # Determinar si hay mirada excesiva basada en la frecuencia de cambios
            if (self.person_data['gaze_change_counter'] >= self.gaze_changes_threshold and
                    elapsed_time <= self.gaze_time_window):
                return True

        return False

    def detect_erratic_movements(self, pose_landmarks, frame_shape, fps):
        """Detecta movimientos erráticos o temblores corporales"""
        if not pose_landmarks:
            return False

        # Puntos clave para seguimiento de estabilidad
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.NOSE
        ]

        # Extraer coordenadas de puntos clave
        current_points = []
        for point_idx in key_points:
            point = pose_landmarks.landmark[point_idx]
            current_points.append([point.x, point.y, point.z if hasattr(point, 'z') else 0])

        current_points = np.array(current_points)
        self.person_data['pose_keypoints_history'].append(current_points)

        # Necesitamos al menos 3 frames para calcular aceleración y jerk
        if len(self.person_data['pose_keypoints_history']) < 3:
            return False

        # Calcular velocidades y aceleraciones
        prev_points = self.person_data['pose_keypoints_history'][-2]
        prev_prev_points = self.person_data['pose_keypoints_history'][-3]

        # Velocidad: distancia entre puntos consecutivos
        velocity = np.linalg.norm(current_points - prev_points, axis=1) * fps
        prev_velocity = np.linalg.norm(prev_points - prev_prev_points, axis=1) * fps

        # Aceleración: cambio en velocidad
        acceleration = np.abs(velocity - prev_velocity) * fps

        # Jerk: cambio en aceleración (derivada de la aceleración)
        if len(self.person_data['pose_keypoints_history']) >= 4:
            prev_prev_prev_points = self.person_data['pose_keypoints_history'][-4]
            prev_prev_velocity = np.linalg.norm(prev_prev_points - prev_prev_prev_points, axis=1) * fps
            prev_acceleration = np.abs(prev_velocity - prev_prev_velocity) * fps
            jerk = np.abs(acceleration - prev_acceleration) * fps

            # Guardar valor máximo de jerk
            max_jerk = np.max(jerk)
            self.person_data['jerk_values'].append(max_jerk)

            # Si el jerk supera el umbral, podría indicar un movimiento brusco
            erratic_movement = max_jerk > self.jerk_threshold
        else:
            erratic_movement = False

        # Análisis de estabilidad (detección de temblores)
        if len(self.person_data['pose_keypoints_history']) >= 5:
            # Calcular oscilaciones en ventanas cortas (indicativo de temblores)
            window_size = 5
            window = list(self.person_data['pose_keypoints_history'])[-window_size:]

            # Calcular varianza de posición para cada punto clave
            variances = []
            for i in range(len(key_points)):
                point_positions = np.array([frame[i] for frame in window])
                variance = np.var(point_positions, axis=0)
                mean_variance = np.mean(variance[:2])  # Solo usar x e y
                variances.append(mean_variance)

            # Verificar si hay oscilaciones pequeñas pero rápidas (temblores)
            max_variance = np.max(variances)

            # Calculamos también la diferencia entre frames consecutivos para detectar oscilaciones
            oscillations = []
            for i in range(1, len(window)):
                diff = np.abs(window[i] - window[i - 1])
                oscillations.append(np.mean(diff))

            mean_oscillation = np.mean(oscillations)

            # Detectar temblor si hay pequeñas oscilaciones consistentes
            has_tremor = (max_variance < self.stability_threshold and
                          mean_oscillation > self.tremor_amplitude_threshold)

            if has_tremor:
                self.person_data['tremor_detected_frames'] += 1
            else:
                self.person_data['tremor_detected_frames'] = 0

            # Considerar temblor si se mantiene por varios frames
            tremor_detected = self.person_data['tremor_detected_frames'] >= self.tremor_frames_threshold
        else:
            tremor_detected = False

        # Combinar detecciones
        return erratic_movement or tremor_detected


    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Configurar video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        frame_counter = 0  # Contador para controlar el salto de frames
        detections = []

        # Ajustar umbrales para compensar frames saltados
        if self.frame_skip > 0:
            self.hidden_hands_frame_threshold = max(1, self.hidden_hands_frame_threshold // self.frame_skip)
            self.tremor_frames_threshold = max(1, self.tremor_frames_threshold // self.frame_skip)

        # Procesamiento frame por frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Incrementar contador de frames
            frame_counter += 1
            frame_idx += 1

            # Procesar solo cada N frames según frame_skip (pero escribir todos los frames)
            #process_this_frame = (frame_counter % self.frame_skip == 0)
            process_this_frame = (self.frame_skip == 0) or (frame_counter % self.frame_skip == 0)
            
            # Crear una copia del frame para escribir al video de salida
            output_frame = frame.copy()

            # Si toca procesar este frame
            if process_this_frame:
                # Convertir BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detectar poses, manos y cara
                pose_results = self.pose.process(frame_rgb)
                hand_results = self.hands.process(frame_rgb)
                face_results = self.face_mesh.process(frame_rgb)

                # Tiempo actual del video
                current_time = frame_idx / fps

                # Variables para comportamientos detectados en este frame
                behaviors = {
                    'hidden_hands': False,
                    'excessive_gaze': False,
                    'erratic_movements': False
                }

                # Procesar si se detectó una persona
                if pose_results.pose_landmarks:
                    # Extraer coordenadas del cuerpo central
                    torso_landmarks = [
                        pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                        pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                        pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                        pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    ]

                    # Calcular posición central
                    current_x = sum(l.x for l in torso_landmarks) / len(torso_landmarks)
                    current_y = sum(l.y for l in torso_landmarks) / len(torso_landmarks)
                    position = (current_x, current_y)

                    # 1. Detectar ocultación de manos o manipulación de bolsillos
                    multi_hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
                    hands_in_pockets = self.detect_hand_pockets(pose_results.pose_landmarks, multi_hand_landmarks,
                                                                frame.shape)

                    if hands_in_pockets:
                        # Incrementar considerando los frames saltados
                        self.person_data['hidden_hands_frames'] += self.frame_skip
                        self.person_data['hidden_hands_duration'] = self.person_data['hidden_hands_frames'] / fps

                        if 'hidden_hands' not in self.person_data['suspicious_start_times']:
                            self.person_data['suspicious_start_times']['hidden_hands'] = current_time
                            self.person_data['hidden_hands_position'] = position

                        # Verificar si ha pasado suficiente tiempo con manos ocultas
                        if (self.person_data['hidden_hands_frames'] > self.hidden_hands_frame_threshold and
                                self.person_data['hidden_hands_duration'] >= self.hidden_hands_time_threshold):
                            behaviors['hidden_hands'] = True

                            if 'hidden_hands' not in self.person_data['alerted']:
                                self.person_data['alerted'].add('hidden_hands')

                            h, w, _ = frame.shape
                            pos_x, pos_y = int(position[0] * w), int(position[1] * h)
                            cv2.putText(output_frame, "Manos ocultas",
                                      (pos_x - 60, pos_y - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.person_data['hidden_hands_frames'] = 0
                        self.person_data['hidden_hands_duration'] = 0
                        if 'hidden_hands' in self.person_data['suspicious_start_times']:
                            del self.person_data['suspicious_start_times']['hidden_hands']
                        if 'hidden_hands' in self.person_data['alerted']:
                            self.person_data['alerted'].remove('hidden_hands')

                    # 2. Detectar miradas excesivas
                    face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
                    excessive_gaze = self.detect_excessive_gaze(face_landmarks, pose_results.pose_landmarks,
                                                              frame.shape, current_time)

                    if excessive_gaze:
                        behaviors['excessive_gaze'] = True

                        if 'excessive_gaze' not in self.person_data['alerted']:
                            self.person_data['alerted'].add('excessive_gaze')

                        h, w, _ = frame.shape
                        pos_x, pos_y = int(position[0] * w), int(position[1] * h - 60)
                        cv2.putText(output_frame, "Mirada excesiva",
                                  (pos_x - 60, pos_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 3. Detectar movimientos erráticos o temblores
                    erratic_movement = self.detect_erratic_movements(pose_results.pose_landmarks, frame.shape, fps)

                    if erratic_movement:
                        if 'erratic_movements' not in self.person_data['suspicious_start_times']:
                            self.person_data['suspicious_start_times']['erratic_movements'] = current_time

                        # Verificar duración del comportamiento
                        movement_duration = current_time - self.person_data['suspicious_start_times']['erratic_movements']

                        if movement_duration >= 1.0:  # Al menos 1 segundo de movimientos erráticos
                            behaviors['erratic_movements'] = True

                            if 'erratic_movements' not in self.person_data['alerted']:
                                self.person_data['alerted'].add('erratic_movements')

                            h, w, _ = frame.shape
                            pos_x, pos_y = int(position[0] * w), int(position[1] * h - 90)
                            cv2.putText(output_frame, "Mov. erratico/temblor",
                                      (pos_x - 90, pos_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if 'erratic_movements' in self.person_data['suspicious_start_times']:
                            del self.person_data['suspicious_start_times']['erratic_movements']
                        if 'erratic_movements' in self.person_data['alerted']:
                            self.person_data['alerted'].remove('erratic_movements')

                    # Guardar detecciones para este frame
                    if any(behaviors.values()):
                        detections.append({
                            'timestamp': current_time,
                            'behaviors': [b for b, detected in behaviors.items() if detected]
                        })

                    # Dibujar landmarks para visualización
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS)

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                output_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS)

                    # Dibujar FaceMesh si está disponible
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            # Dibujar solo el contorno facial y los ojos
                            connections = []
                            for connection in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                                connections.append(connection)
                            self.mp_drawing.draw_landmarks(
                                output_frame,
                                face_landmarks,
                                connections,
                                landmark_drawing_spec=None)

                # Mostrar contador de personas en la esquina
                num_people = 1 if pose_results.pose_landmarks else 0
                cv2.putText(output_frame, f"Personas: {num_people}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Escribir frame procesado al video de salida
            out.write(output_frame)
            _, buffer = cv2.imencode('.jpg', output_frame)
            jpg_b64 = base64.b64encode(buffer).decode('utf-8')
            yield jpg_b64

            # Mostrar progreso cada 100 frames
            if frame_idx % 100 == 0:
                print(f"Procesado {frame_idx}/{total_frames} frames ({frame_idx / total_frames * 100:.1f}%)")

        # Liberar recursos
        cap.release()
        out.release()

        yield detections