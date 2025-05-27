from multiprocessing import Process
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import defaultdict, deque
from Resources.Helper import get_temp_route, format_number
import base64
import uuid

class BehaviorDetector3D:
    def __init__(self, frame_skip=3):
        # Inicializar MediaPipe Pose, Hands y FaceMesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
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

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Variables para seguimiento de comportamientos
        self.person_data = {
            'positions': deque(maxlen=30),  # Posiciones 3D recientes (últimos 30 frames)
            'hidden_hands_frames': 0,
            'hidden_hands_duration': 0,
            'hidden_hands_position': None,
            'suspicious_start_times': {},
            'alerted': set(),
            'gaze_directions': deque(maxlen=60),  # Historial de direcciones de mirada en 3D
            'gaze_change_counter': 0,
            'last_significant_gaze_time': 0,
            'proximity_distances': deque(maxlen=30),  # Historial de distancias 3D
            'approach_detected_frames': 0,
            'initial_distances': None,
            'reference_established': False
        }

        # Umbrales ajustados para 3D
        self.hidden_hands_frame_threshold = 25
        self.hidden_hands_time_threshold = 3.0
        self.gaze_angle_threshold = 0.3  # ~17 grados
        self.gaze_changes_threshold = 8
        self.gaze_time_window = 5.0
        self.proximity_threshold = 0.25
        self.approach_frames_threshold = 15
        self.min_detection_frames = 10
        self.confidence_threshold = 0.7

        # Índices para FaceMesh
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 374, 373, 390]
        self.IRIS_INDICES = [468, 469, 470, 471, 472, 473]

    def detect_hand_pockets(self, pose_landmarks, hand_landmarks, frame_shape):
        """Detecta si las manos están en bolsillos o detrás de la espalda usando coordenadas 3D"""
        if not pose_landmarks:
            return False

        h, w, _ = frame_shape

        # Extraer coordenadas 3D
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calcular punto medio de los hombros en 3D
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        ])

        # Verificar manos visibles
        visible_hands = set()
        if hand_landmarks:
            for hand_lm in hand_landmarks:
                # Calcular punto medio de la mano en 3D
                hand_pos = np.array([
                    sum(lm.x for lm in hand_lm.landmark) / len(hand_lm.landmark),
                    sum(lm.y for lm in hand_lm.landmark) / len(hand_lm.landmark),
                    sum(lm.z for lm in hand_lm.landmark) / len(hand_lm.landmark)
                ])
                # Determinar mano izquierda o derecha
                if hand_pos[0] < shoulder_mid[0]:
                    visible_hands.add('left')
                else:
                    visible_hands.add('right')

        # Verificar si las muñecas están cerca de las caderas o detrás en 3D
        hands_in_pockets = False

        # Mano izquierda
        if 'left' not in visible_hands:
            hip_wrist_vec = np.array([left_hip.x - left_wrist.x, left_hip.y - left_wrist.y, left_hip.z - left_wrist.z])
            hip_wrist_distance = np.linalg.norm(hip_wrist_vec)
            behind_back = left_wrist.z > left_hip.z + 0.1
            if hip_wrist_distance < 0.15 or behind_back:
                hands_in_pockets = True

        # Mano derecha
        if 'right' not in visible_hands:
            hip_wrist_vec = np.array([right_hip.x - right_wrist.x, right_hip.y - right_wrist.y, right_hip.z - right_wrist.z])
            hip_wrist_distance = np.linalg.norm(hip_wrist_vec)
            behind_back = right_wrist.z > right_hip.z + 0.1
            if hip_wrist_distance < 0.15 or behind_back:
                hands_in_pockets = True

        return hands_in_pockets

    def detect_excessive_gaze(self, face_landmarks, pose_landmarks, frame_shape, current_time):
        """Detecta mirada excesiva usando vectores 3D"""
        if not face_landmarks and not pose_landmarks:
            return False

        if face_landmarks:
            # Calcular centros de ojos en 3D
            left_eye_center = np.mean([[face_landmarks.landmark[idx].x,
                                      face_landmarks.landmark[idx].y,
                                      face_landmarks.landmark[idx].z]
                                     for idx in self.LEFT_EYE_INDICES], axis=0)
            right_eye_center = np.mean([[face_landmarks.landmark[idx].x,
                                       face_landmarks.landmark[idx].y,
                                       face_landmarks.landmark[idx].z]
                                      for idx in self.RIGHT_EYE_INDICES], axis=0)
            eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)

            # Usar iris si está disponible
            has_iris_data = len(face_landmarks.landmark) > 468
            if has_iris_data:
                left_iris = np.mean([[face_landmarks.landmark[idx].x,
                                    face_landmarks.landmark[idx].y,
                                    face_landmarks.landmark[idx].z]
                                   for idx in self.IRIS_INDICES[:3]], axis=0)
                right_iris = np.mean([[face_landmarks.landmark[idx].x,
                                     face_landmarks.landmark[idx].y,
                                     face_landmarks.landmark[idx].z]
                                    for idx in self.IRIS_INDICES[3:]], axis=0)
                gaze_vector = np.mean([left_iris - left_eye_center, right_iris - right_eye_center], axis=0)
            else:
                nose_tip = np.array([face_landmarks.landmark[4].x,
                                   face_landmarks.landmark[4].y,
                                   face_landmarks.landmark[4].z])
                gaze_vector = nose_tip - eyes_center

        elif pose_landmarks:
            nose = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].z])
            left_eye = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x,
                               pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y,
                               pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].z])
            right_eye = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y,
                                pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].z])
            eyes_center = np.mean([left_eye, right_eye], axis=0)
            gaze_vector = nose - eyes_center

        # Normalizar vector de mirada
        gaze_magnitude = np.linalg.norm(gaze_vector)
        if gaze_magnitude > 0:
            gaze_vector = gaze_vector / gaze_magnitude
        else:
            return False

        self.person_data['gaze_directions'].append(gaze_vector)
        if len(self.person_data['gaze_directions']) < 2:
            return False

        # Comparar con dirección anterior
        prev_gaze = self.person_data['gaze_directions'][-2]
        dot_product = np.clip(np.dot(gaze_vector, prev_gaze), -1.0, 1.0)
        angle_change = np.arccos(dot_product)

        if angle_change > self.gaze_angle_threshold:
            self.person_data['gaze_change_counter'] += 1
            self.person_data['last_significant_gaze_time'] = current_time
            if 'excessive_gaze' not in self.person_data['suspicious_start_times']:
                self.person_data['suspicious_start_times']['excessive_gaze'] = current_time

        if 'excessive_gaze' in self.person_data['suspicious_start_times']:
            elapsed_time = current_time - self.person_data['suspicious_start_times']['excessive_gaze']
            if current_time - self.person_data['last_significant_gaze_time'] > self.gaze_time_window:
                self.person_data['gaze_change_counter'] = 0
                del self.person_data['suspicious_start_times']['excessive_gaze']
                if 'excessive_gaze' in self.person_data['alerted']:
                    self.person_data['alerted'].remove('excessive_gaze')
                return False

            if (self.person_data['gaze_change_counter'] >= self.gaze_changes_threshold and
                    elapsed_time <= self.gaze_time_window):
                return True

        return False

    def detect_camera_approach(self, pose_landmarks, frame_shape):
        """Detecta acercamiento a la cámara usando dimensiones 3D"""
        if not pose_landmarks:
            return False

        # Puntos clave en 3D
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.NOSE
        ]

        # Calcular dimensiones 3D
        shoulder_vec = np.array([
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x - 
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y - 
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z - 
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z
        ])
        hip_vec = np.array([
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x - 
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y - 
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z - 
            pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z
        ])
        torso_height_vec = np.array([
            (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
             pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 - 
            (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x + 
             pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
            (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
             pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 - 
            (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y + 
             pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2,
            (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z + 
             pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2 - 
            (pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z + 
             pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z) / 2
        ])

        shoulder_width = np.linalg.norm(shoulder_vec)
        hip_width = np.linalg.norm(hip_vec)
        torso_height = np.linalg.norm(torso_height_vec)
        apparent_size = (shoulder_width + hip_width) * torso_height

        self.person_data['proximity_distances'].append(apparent_size)
        if len(self.person_data['proximity_distances']) < self.min_detection_frames:
            return False

        if not self.person_data['reference_established']:
            initial_distances = list(self.person_data['proximity_distances'])[:self.min_detection_frames]
            self.person_data['initial_distances'] = np.mean(initial_distances)
            self.person_data['reference_established'] = True
            return False

        current_size = self.person_data['proximity_distances'][-1]
        reference_size = self.person_data['initial_distances']
        size_change_ratio = current_size / reference_size if reference_size > 0 else 1.0

        recent_trend = False
        if len(self.person_data['proximity_distances']) >= 5:
            recent_sizes = list(self.person_data['proximity_distances'])[-5:]
            if all(recent_sizes[i] >= recent_sizes[i-1] for i in range(1, len(recent_sizes))):
                recent_trend = True

        approaching = size_change_ratio > (1 + self.proximity_threshold) and recent_trend
        if approaching:
            self.person_data['approach_detected_frames'] += 1
        else:
            self.person_data['approach_detected_frames'] = max(0, self.person_data['approach_detected_frames'] - 1)

        return self.person_data['approach_detected_frames'] >= self.approach_frames_threshold

    def process_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_idx = 0
            frame_counter = 0
            detections = []

            if self.frame_skip > 0:
                self.hidden_hands_frame_threshold = max(1, self.hidden_hands_frame_threshold // self.frame_skip)
                self.approach_frames_threshold = max(1, self.approach_frames_threshold // self.frame_skip)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                frame_idx += 1
                process_this_frame = (self.frame_skip == 0) or (frame_counter % self.frame_skip == 0)
                output_frame = frame.copy()

                if process_this_frame:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(frame_rgb)
                    hand_results = self.hands.process(frame_rgb)
                    face_results = self.face_mesh.process(frame_rgb)
                    current_time = frame_idx / fps

                    behaviors = {
                        'hidden_hands': False,
                        'excessive_gaze': False,
                        'camera_approach': False
                    }

                    if pose_results.pose_landmarks:
                        # Calcular posición central en 3D
                        torso_landmarks = [
                            pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                            pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                            pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                            pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
                        ]
                        current_pos = np.array([
                            sum(l.x for l in torso_landmarks) / len(torso_landmarks),
                            sum(l.y for l in torso_landmarks) / len(torso_landmarks),
                            sum(l.z for l in torso_landmarks) / len(torso_landmarks)
                        ])
                        position = (current_pos[0], current_pos[1])  # Mantener 2D para visualización

                        # Detectar manos ocultas
                        multi_hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
                        hands_in_pockets = self.detect_hand_pockets(pose_results.pose_landmarks, multi_hand_landmarks, frame.shape)

                        if hands_in_pockets:
                            self.person_data['hidden_hands_frames'] += self.frame_skip
                            self.person_data['hidden_hands_duration'] = self.person_data['hidden_hands_frames'] / fps
                            if 'hidden_hands' not in self.person_data['suspicious_start_times']:
                                self.person_data['suspicious_start_times']['hidden_hands'] = current_time
                                self.person_data['hidden_hands_position'] = position

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

                        # Detectar mirada excesiva
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

                        # Detectar acercamiento
                        camera_approach = self.detect_camera_approach(pose_results.pose_landmarks, frame.shape)

                        if camera_approach:
                            if 'camera_approach' not in self.person_data['suspicious_start_times']:
                                self.person_data['suspicious_start_times']['camera_approach'] = current_time
                            approach_duration = current_time - self.person_data['suspicious_start_times']['camera_approach']
                            if approach_duration >= 1.0:
                                behaviors['camera_approach'] = True
                                if 'camera_approach' not in self.person_data['alerted']:
                                    self.person_data['alerted'].add('camera_approach')
                                h, w, _ = frame.shape
                                pos_x, pos_y = int(position[0] * w), int(position[1] * h - 90)
                                cv2.putText(output_frame, "Acercamiento sospechoso",
                                        (pos_x - 90, pos_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            if 'camera_approach' in self.person_data['suspicious_start_times']:
                                del self.person_data['suspicious_start_times']['camera_approach']
                            if 'camera_approach' in self.person_data['alerted']:
                                self.person_data['alerted'].remove('camera_approach')

                        if any(behaviors.values()):
                            detections.append({
                                'timestamp': current_time,
                                'behaviors': [b for b, detected in behaviors.items() if detected]
                            })

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

                        if face_results.multi_face_landmarks:
                            for face_landmarks in face_results.multi_face_landmarks:
                                connections = []
                                for connection in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                                    connections.append(connection)
                                self.mp_drawing.draw_landmarks(
                                    output_frame,
                                    face_landmarks,
                                    connections,
                                    landmark_drawing_spec=None)

                    num_people = 1 if pose_results.pose_landmarks else 0
                    cv2.putText(output_frame, f"Personas: {num_people}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', output_frame)
                jpg_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data = {
                    'frame': jpg_b64,
                    'progress': f"{round(frame_idx / total_frames * 100)}",
                    'detections': list(detections[-1]['behaviors']) if detections else None
                }

                yield frame_data
                out.write(output_frame)

            yield detections
        finally:
            cap.release()
            out.release()