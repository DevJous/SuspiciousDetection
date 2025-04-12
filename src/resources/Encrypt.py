import hashlib

def encrypt_password(password):
    password_bytes = password.encode('utf-8')
    hashed_password = hashlib.sha256(password_bytes).hexdigest()

    return hashed_password

