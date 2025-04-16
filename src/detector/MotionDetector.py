import cv2

class MotionDetector:
    def __init__(self):
        self.prev_frame = None
        self.min_contour_area = 500  # Área mínima para considerar movimiento

    def detect_suspicious_behavior(self, frame):
        # Convertir a escala de grises y aplicar desenfoque
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return []

        # Calcular diferencia entre frames
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        suspicious_boxes = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            # Obtener coordenadas del rectángulo
            (x, y, w, h) = cv2.boundingRect(contour)
            suspicious_boxes.append((x, y, x + w, y + h))

        self.prev_frame = gray
        return suspicious_boxes