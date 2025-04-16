import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import os

"""
.onnx para onnxruntime
.pt para PyTorch
"""

model = YOLO("models/yolov8m.pt")
model.export(format="onnx")

class YoloDetector:
    def __init__(self, model_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..", "models", "yolov8m.onnx")), conf_thres=0.5, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.initialize_model(model_path)

        # Nombres de clases COCO (puedes personalizar esto)
        self.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                            "scissors", "teddy bear", "hair drier", "toothbrush"]

    def initialize_model(self, model_path):
        # Verificar si el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontro el archivo en la ruta: {model_path}. Primero descarguelo.")
        # Sesión ONNX Runtime
        self.session = ort.InferenceSession(model_path)

        # Obtener información del modelo
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Realizar inferencia
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Procesar salidas
        boxes, scores, class_ids = self.process_output(outputs)

        return boxes, scores, class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        # Redimensionar y normalizar
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Escalar píxeles a [0, 1]
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filtrar por confianza
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Obtener clase con máxima confianza
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Obtener cajas delimitadoras
        boxes = self.extract_boxes(predictions)

        # Aplicar NMS
        indices = self.nms(boxes, scores)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)

        # Convertir de xywh a xyxy
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        boxes_[..., 2] = boxes[..., 0] + boxes[..., 2] * 0.5
        boxes_[..., 3] = boxes[..., 1] + boxes[..., 3] * 0.5

        return boxes_

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        img_shape = np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= img_shape
        return boxes

    def nms(self, boxes, scores):
        indices = np.argsort(scores)[::-1]
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            iou = self.compute_iou(current_box, remaining_boxes)
            remaining_indices = np.where(iou <= self.iou_threshold)[0]
            indices = indices[remaining_indices + 1]

        return np.array(keep)

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        return intersection_area / union_area

    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            color = (0, 255, 0)  # Verde para objetos
            if class_id == 0:  # Rojo para personas
                color = (0, 0, 255)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{self.class_names[class_id]}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image