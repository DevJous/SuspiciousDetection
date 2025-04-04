from flask import request, json, jsonify, g
import jwt
from functools import wraps

SECRET_KEY  = "@JustD0I7_2024X"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            print('auth_header:',auth_header)
            return jsonify({'authenticated': False, 'message': 'Token es requerido'}), 401

        try:
            scheme, token = auth_header.split()
            if scheme.lower() != 'bearer':
                raise ValueError('Formato de encabezado incorrecto')
        except ValueError:
            return jsonify({'authenticated': False, 'message': 'Token malformado'}), 401

        try:
            decoded_token = jwt.decode(token, get_key(), algorithms=["HS256"])
            g.user_id = decoded_token.get('id')
        except jwt.ExpiredSignatureError:
            return jsonify({'authenticated': False, 'message': 'Token ha expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'authenticated': False, 'message': 'Token inválido'}), 401

        return f(*args, **kwargs)
    return decorated


def get_key():
    return SECRET_KEY

#def deserialize_token():
#    decoded_data = jwt.decode(token, SECRET_KEY , algorithms=["HS256"])
#    print(decoded_data)
#    return decoded_data