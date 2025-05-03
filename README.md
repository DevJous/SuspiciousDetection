# Sistema de Detección de Posturas y Análisis de Comportamientos

Este proyecto implementa un sistema basado en Flask y MediaPipe para la detección de posturas humanas y análisis de comportamientos sospechosos mediante visión por computadora.

## Características Principales

- 🖼️ **Análisis de Imágenes**: Detecta keypoints de postura humana en imágenes y genera datasets en formato JSON con las coordenadas.
- 🎬 **Análisis de Videos**: Procesa videos para detectar comportamientos sospechosos, marcando el tiempo y categorizando los eventos.
- 📷 **Detección en Tiempo Real**: Utiliza la cámara del dispositivo para detectar comportamientos sospechosos basados en la postura en tiempo real.

## Instalación

### Primer paso
- Instala las librerias que se encuentran dentro del archivo requirements.txt para poder inicializar el proyecto

```bash
pip install -r requirements.txt
```

### Requisitos del Sistema

- Python 3.12.4
- Flask
- MediaPipe
- OpenCV
- Navegador web moderno compatible con JavaScript (Chrome, Edge, Opera, Firefox)

## Uso

### Iniciar la Aplicación

```bash
python app.py
```

Tras ejecutarse, la aplicación estará disponible en `http://localhost:5000` (o el puerto especificado).
El proyecto actualmente se encuentra hosteado en `https://devjous.site` para su uso demostrativo.

### Interfaz Web

La interfaz principal permite:

1. **Cargar Imágenes**: 
   - Sube una imagen para analizar posturas humanas
   - Visualiza los keypoints detectados
   - Descarga los datos en formato JSON

2. **Cargar Videos**:
   - Sube un video para analizar comportamientos
   - Visualiza marcadores temporales de comportamientos sospechosos
   - Obtén un informe categorizado de eventos detectados

3. **Detección en Tiempo Real**:
   - Permite el acceso a la cámara
   - Muestra los análisis de postura en tiempo real
   - Alerta de comportamientos sospechosos

## Estructura del Proyecto

```
proyecto-flask-mediapipe/
│
├── main.py                  # Aplicación principal de Flask
├── requirements.txt         # Dependencias del proyecto
├── Resources/               # Archivos base del proyeco
├── static/                  # Archivos estáticos
│   ├── assets/              # Recursos de imagen o video
│   ├── scripts/             # Scripts JavaScript 
│   ├── style/               # Estilos por plantilla html
│   └── style.css            # Imágenes del sistema
├── templates/               # Plantillas HTML
├── .env                     # Arhivo con data sensible
└── README.md
```

## Formato de Datos JSON

El sistema genera archivos JSON con la siguiente estructura para los keypoints:

```json
{
  "timestamp": "2023-05-03T14:30:00",
  "keypoints": [
    {"id": 0, "name": "nose", "x": 0.5, "y": 0.3, "z": 0.2, "visibility": 0.98},
    {"id": 1, "name": "left_eye", "x": 0.45, "y": 0.27, "z": 0.21, "visibility": 0.96},
    // ... otros keypoints
  ],
  "confidence": 0.89
}
```

## Categorías de Comportamientos

El sistema detecta y clasifica comportamientos en las siguientes categorías:

- **Caídas**: Detecta cambios bruscos en la posición vertical
- **Comportamiento Errático**: Movimientos rápidos e impredecibles
- **Posturas Inusuales**: Configuraciones corporales fuera de lo común
- **Proximidad Sospechosa**: Detección de acercamientos anómalos

## Contribución

1. Haz un fork del proyecto
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva característica'`)
4. Sube tus cambios (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Reconocimientos

- [MediaPipe](https://google.github.io/mediapipe/) por el framework de detección de posturas
- [Flask](https://flask.palletsprojects.com/) por el framework web
- [OpenCV](https://opencv.org/) por el procesamiento de imágenes y video