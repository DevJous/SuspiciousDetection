import cv2
import numpy as np
from src.detector.MotionDetector import MotionDetector
from src.detector.YoloDetector import YoloDetector

class CameraController:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.yolo_detector = YoloDetector()
        self.suspicious_behaviors = {
            "loitering": "Merodeo",
            "fighting": "Pelea",
            "stealing": "Robo",
            "intrusion": "Intrusión",
            "vandalism": "Vandalismo"
        }

    def process_frame_from_bytes(self, img_bytes):
        # Decodificar imagen a formato OpenCV
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        # Detección con YOLO
        boxes, scores, class_ids = self.yolo_detector.detect_objects(frame)

        # Detección de movimiento
        suspicious_areas = self.motion_detector.detect_suspicious_behavior(frame)

        # Dibujar resultados
        processed_frame = self._process_frame(frame, boxes, scores, class_ids, suspicious_areas)

        return processed_frame

    def _process_frame(self, frame, boxes, scores, class_ids, suspicious_areas):
        frame = self.yolo_detector.draw_detections(frame, boxes, scores, class_ids)

        for (x1, y1, x2, y2) in suspicious_areas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            behavior = "loitering"  # Ejemplo estático
            label = f"Sospechoso: {self.suspicious_behaviors.get(behavior, 'Inusual')}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame
