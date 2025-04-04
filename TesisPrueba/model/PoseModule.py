import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self,
                 mode=False,
                 modelComplexity=1,
                 smooth=True,
                 detectionCon=0.5,
                 trackCon=0.5 ):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.modelComplexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB porque MediaPipe trabaja en RGB
        self.results = self.pose.process(imgRGB)  # Procesar la imagen para encontrar poses
        lmList = []

        if self.results.pose_landmarks:  # Si hay detección de landmarks
            h, w, c = img.shape  # Obtener las dimensiones de la imagen

            # Filtrar y procesar los landmarks excluyendo los puntos que no deseas
            landmarks = [
                (id, lm) for id, lm in enumerate(self.results.pose_landmarks.landmark)
                if id not in [1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]
            ]

            # Imprimir los puntos restantes
            for id, lm in landmarks:
                lmList.append([id, int(lm.x * w), int(lm.y * h)])

            # Calcular el centro del pecho usando los hombros (IDs 11 y 12)
            shoulder_left = next(lm for lm in lmList if lm[0] == 11)
            shoulder_right = next(lm for lm in lmList if lm[0] == 12)

            # Calcular las coordenadas del centro del pecho
            chest_x = (shoulder_left[1] + shoulder_right[1]) // 2
            chest_y = (shoulder_left[2] + shoulder_right[2]) // 2

            # Añadir el centro del pecho al lmList con un ID personalizado
            lmList.append([33, chest_x, chest_y])  # 33 es un ID arbitrario para el pecho

            for lm in lmList:
                x, y = lm[1], lm[2]
                #cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)

            # Lista de conexiones entre puntos (IDs) que deseas conectar
            connections = [
                (11, 12),  # Hombros izquierdo y derecho
                (11, 13),  # Hombro izquierdo a codo izquierdo
                (13, 15),  # Codo izquierdo a muñeca izquierda
                (12, 14),  # Hombro derecho a codo derecho
                (14, 16),  # Codo derecho a muñeca derecha
                (11, 23),  # Hombro izquierdo a cadera izquierda
                (12, 24),  # Hombro derecho a cadera derecha
                (23, 24),  # Cadera izquierda a cadera derecha
                (23, 25),  # Cadera izquierda a rodilla izquierda
                (25, 27),  # Rodilla izquierda a tobillo izquierdo
                (24, 26),  # Cadera derecha a rodilla derecha
                (26, 28),  # Rodilla derecha a tobillo derecho
                (11, 33),  # Hombro izquierdo al centro del pecho
                (12, 33),  # Hombro derecho al centro del pecho
                (33, 0 ),  # Conectar pecho con nariz
            ]

            # Dibuja las conexiones en la imagen
            for (start_id, end_id) in connections:
                # Encuentra los puntos en lmList usando sus IDs
                start_point = next((x, y) for id, x, y in lmList if id == start_id)
                end_point = next((x, y) for id, x, y in lmList if id == end_id)

                #if start_point and end_point:
                #    cv2.line(img, start_point, end_point, (171, 231, 0), 1)  # Color azul y grosor 2

        return img

    def findPosition(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB porque MediaPipe trabaja en RGB
        self.results = self.pose.process(imgRGB)  # Procesar la imagen para encontrar poses
        lmList = []

        if self.results.pose_landmarks:  # Si hay detección de landmarks
            h, w, c = img.shape  # Obtener las dimensiones de la imagen

            # Filtrar y procesar los landmarks excluyendo los puntos que no deseas
            landmarks = [
                (id, lm) for id, lm in enumerate(self.results.pose_landmarks.landmark)
                if id not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]
            ]

            # Imprimir los puntos restantes
            for id, lm in landmarks:
                lmList.append([id, int(lm.x * w), int(lm.y * h)])

            # Calcular el centro del pecho usando los hombros (IDs 11 y 12)
            shoulder_left = next(lm for lm in lmList if lm[0] == 11)
            shoulder_right = next(lm for lm in lmList if lm[0] == 12)

            # Calcular las coordenadas del centro del pecho
            chest_x = (shoulder_left[1] + shoulder_right[1]) // 2
            chest_y = (shoulder_left[2] + shoulder_right[2]) // 2

            # Añadir el centro del pecho al lmList con un ID personalizado
            lmList.append([33, chest_x, chest_y])  # 33 es un ID arbitrario para el pecho

            for lm in lmList:
                x, y = lm[1], lm[2]
                #cv2.circle(img, (x, y), 5, (0, 0, 0), cv2.FILLED)
        return lmList
