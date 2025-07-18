import os
import platform
import string

# Esto es modificable, a efectos de prueba se usa la ruta "Escritorio" de Windows y la
# ruta "home" de Linux (dependiendo del SO detectado), pero no se recomienda para producción
def get_work_path():
    if platform.system() == 'Windows':
        try:
            import winreg
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            desktop_path, _ = winreg.QueryValueEx(reg_key, "Desktop")
            winreg.CloseKey(reg_key)
            return os.path.join(desktop_path, "SospiciousDetection")
        except:
            return os.path.join(os.environ["USERPROFILE"], "Desktop", "SospiciousDetection")
    else:
        return os.path.abspath(os.path.join(os.path.expanduser("~"), "..", "home", "SospiciousDetection"))

def get_temp_route(child=None):
    if platform.system() == 'Windows':
        import winreg
        try:
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            desktop_path, _ = winreg.QueryValueEx(reg_key, "Desktop")
            winreg.CloseKey(reg_key)
            return os.path.join(desktop_path, "SospiciousDetection", "temp", child) if child else os.path.join(desktop_path, "SospiciousDetection", "temp")
        except:
            return os.path.join(os.environ["USERPROFILE"], "Desktop", "SospiciousDetection", "temp", child) if child else os.path.join(os.environ["USERPROFILE"], "Desktop", "SospiciousDetection", "temp")
    else:
        return os.path.abspath(os.path.join(os.path.expanduser("~"), "..", "home", "SospiciousDetection", "temp", child)) if child else os.path.abspath(os.path.join(os.path.expanduser("~"), "..", "home", "SospiciousDetection", "temp"))
    
def get_processed_route(child=None):
    if platform.system() == 'Windows':
        import winreg
        try:
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            desktop_path, _ = winreg.QueryValueEx(reg_key, "Desktop")
            winreg.CloseKey(reg_key)
            return os.path.join(desktop_path, "SospiciousDetection", "Processeds", child) if child else os.path.join(desktop_path, "SospiciousDetection", "Processeds")
        except:
            return os.path.join(os.environ["USERPROFILE"], "Desktop", "SospiciousDetection", "Processeds", child) if child else os.path.join(os.environ["USERPROFILE"], "Desktop", "SospiciousDetection", "Processeds")
    else:
        return os.path.abspath(os.path.join(os.path.expanduser("~"), "..", "home", "SospiciousDetection", "Processeds", child)) if child else os.path.abspath(os.path.join(os.path.expanduser("~"), "..", "home", "SospiciousDetection", "Processeds"))
    
def format_number(num, length=6):
    return str(num).zfill(length)

def normalizeUrl(url:str):
    return url.replace("{slash}", "/") if url != 'null' else None