from multiprocessing import process
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import defaultdict, deque
import base64

class BehaviorDetector3D:
    def __init__(self, frame_skip=3, connection=None, with_camera=False):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.frame_skip = frame_skip
        self.with_camera = with_camera
        self.connection = connection

        # Configuración optimizada
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Estructura de datos
        self.person_data = {
            'positions_3d': deque(maxlen=30),
            'hidden_hands_frames': 0,
            'hidden_hands_duration': 0,
            'hidden_hands_position_3d': None,
            'suspicious_start_times': {},
            'alerted': set(),
            'gaze_directions_3d': deque(maxlen=60),
            'gaze_change_counter': 0,
            'last_significant_gaze_time': 0,
            'hand_under_clothes_frames': 0,
            'hand_under_clothes_duration': 0,
            'velocity_3d': deque(maxlen=10),
            'acceleration_3d': deque(maxlen=10),
            'hand_depth_history': deque(maxlen=15),
            'torso_depth_reference': deque(maxlen=10),
            'body_orientation': deque(maxlen=5),
        }

        # Parámetros optimizados
        self.hidden_hands_frame_threshold = 20
        self.hidden_hands_time_threshold = 2.5
        self.gaze_angle_threshold_3d = 0.35
        self.gaze_changes_threshold = 6
        self.gaze_time_window = 4.0
        self.hand_under_clothes_frame_threshold = 15
        self.hand_under_clothes_time_threshold = 2
        self.arm_angle_threshold_min = 100
        self.arm_angle_threshold_max = 140
        self.confidence_threshold = 0.65
        
        # Umbrales de profundidad ajustados
        self.depth_threshold_behind = 0.08  # Manos claramente detrás del torso
        self.depth_threshold_front = 0.05   # Manos claramente delante del torso
        self.depth_consistency_threshold = 0.65
        self.torso_depth_variance_threshold = 0.025
        
        self.movement_threshold = 0.015
        self.suspicious_velocity_threshold = 1.2

        # Índices faciales
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 374, 373, 390]
        self.IRIS_INDICES = [468, 469, 470, 471, 472, 473]

    def get_3d_position(self, landmark):
        """Obtiene la posición 3D de un landmark"""
        if landmark is None:
            return None
        return np.array([landmark.x, landmark.y, landmark.z])

    def calculate_3d_distance(self, pos1, pos2):
        """Calcula la distancia euclidiana entre dos puntos 3D"""
        if pos1 is None or pos2 is None:
            return float('inf')
        return np.linalg.norm(pos1 - pos2)

    def calculate_3d_angle(self, v1, v2):
        """Calcula el ángulo entre dos vectores 3D en grados"""
        if v1 is None or v2 is None:
            return 0.0
            
        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_body_orientation(self, pose_landmarks):
        """Calcula la orientación del cuerpo en el espacio 3D"""
        if pose_landmarks is None:
            return None, None, None
            
        left_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        left_hip = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        
        torso_vector = ((left_hip + right_hip) / 2) - ((left_shoulder + right_shoulder) / 2)
        shoulder_vector = right_shoulder - left_shoulder
        
        body_normal = np.cross(torso_vector, shoulder_vector)
        body_normal = body_normal / np.linalg.norm(body_normal) if np.linalg.norm(body_normal) > 0 else body_normal
        
        return body_normal, torso_vector, shoulder_vector

    def get_torso_depth_reference(self, pose_landmarks):
        """Obtiene la profundidad de referencia del torso"""
        if pose_landmarks is None:
            return 0.0
            
        chest_landmarks = [
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]),
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        ]
        
        torso_depth = np.mean([pos[2] for pos in chest_landmarks if pos is not None])
        self.person_data['torso_depth_reference'].append(torso_depth)
        
        return torso_depth

    def analyze_hand_depth_position(self, pose_landmarks, hand_landmarks):
        """Analiza la posición de profundidad de las manos con separación clara entre delante/detrás"""
        if pose_landmarks is None:
            return {
                'left_hand': {'behind': False, 'front': False, 'depth_confidence': 0},
                'right_hand': {'behind': False, 'front': False, 'depth_confidence': 0},
                'torso_depth': 0.0,
                'consistency': {
                    'behind': {'left': 0.0, 'right': 0.0},
                    'front': {'left': 0.0, 'right': 0.0}
                }
            }

        torso_depth = self.get_torso_depth_reference(pose_landmarks)
        
        # Obtener posiciones de las muñecas y hombros
        left_wrist = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST])
        right_wrist = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        left_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        
        # Verificar visibilidad de manos
        visible_hands = {'left': False, 'right': False}
        hand_depths = {'left': None, 'right': None}
        
        if hand_landmarks:
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            for hand_lm in hand_landmarks:
                hand_center = np.mean([self.get_3d_position(lm) for lm in hand_lm.landmark], axis=0)
                if hand_center[0] < shoulder_mid[0]:  # Mano izquierda
                    visible_hands['left'] = True
                    hand_depths['left'] = hand_center[2]
                else:  # Mano derecha
                    visible_hands['right'] = True
                    hand_depths['right'] = hand_center[2]

        def analyze_single_hand(wrist_pos, shoulder_pos, hand_side, is_visible, hand_depth):
            result = {'behind': False, 'front': False, 'depth_confidence': 0}
            
            if wrist_pos is None or shoulder_pos is None:
                return result
                
            wrist_depth_diff = wrist_pos[2] - torso_depth
            
            # MANOS OCULTAS (en bolsillos) - SOLO cuando is_visible = False
            if not is_visible:
                if wrist_depth_diff > self.depth_threshold_behind:
                    result['behind'] = True
                    result['depth_confidence'] = min(1.0, wrist_depth_diff / self.depth_threshold_behind)
            
            # MANOS VISIBLES (bajo ropa) - SOLO cuando is_visible = True
            else:
                if hand_depth is not None and (hand_depth - torso_depth) < -self.depth_threshold_front:
                    result['front'] = True
                    result['depth_confidence'] = min(1.0, abs(hand_depth - torso_depth) / self.depth_threshold_front)
            
            return result

        # Analizar ambas manos
        left_analysis = analyze_single_hand(
            left_wrist, left_shoulder, 'left', 
            visible_hands['left'], hand_depths['left']
        )
        
        right_analysis = analyze_single_hand(
            right_wrist, right_shoulder, 'right', 
            visible_hands['right'], hand_depths['right']
        )
        
        # Guardar en historial para consistencia temporal
        hand_state = {
            'left_behind': left_analysis['behind'],
            'left_front': left_analysis['front'],
            'right_behind': right_analysis['behind'],
            'right_front': right_analysis['front'],
            'timestamp': len(self.person_data['hand_depth_history'])
        }
        
        self.person_data['hand_depth_history'].append(hand_state)
        
        return {
            'left_hand': left_analysis,
            'right_hand': right_analysis,
            'torso_depth': torso_depth,
            'consistency': self.calculate_depth_consistency()
        }

    def calculate_depth_consistency(self):
        """Calcula la consistencia de la detección de profundidad en frames recientes"""
        if len(self.person_data['hand_depth_history']) < 5:
            return {
                'behind': {'left': 0.0, 'right': 0.0},
                'front': {'left': 0.0, 'right': 0.0}
            }
        
        recent_states = list(self.person_data['hand_depth_history'])[-5:]
        
        # Consistencia para manos detrás
        behind_consistency = {
            'left': sum(1 for state in recent_states if state['left_behind']) / len(recent_states),
            'right': sum(1 for state in recent_states if state['right_behind']) / len(recent_states)
        }
        
        # Consistencia para manos delante
        front_consistency = {
            'left': sum(1 for state in recent_states if state['left_front']) / len(recent_states),
            'right': sum(1 for state in recent_states if state['right_front']) / len(recent_states)
        }
        
        return {
            'behind': behind_consistency,
            'front': front_consistency
        }

    def detect_hand_pockets_3d(self, pose_landmarks, hand_landmarks, frame_shape):
        """Detección mejorada de manos en bolsillos (SOLO DETRÁS del cuerpo)"""
        if pose_landmarks is None:
            return False
        
        # Verificar primero si las manos están realmente ocultas
        hands_visible = hand_landmarks is not None and len(hand_landmarks) > 0
        
        # Si las manos son visibles, NO pueden estar en bolsillos
        if hands_visible:
            return False

        # Solo analizar profundidad si las manos están ocultas
        depth_analysis = self.analyze_hand_depth_position(pose_landmarks, None)  # Pasar None para manos ocultas
        consistency = depth_analysis.get('consistency', {
            'behind': {'left': 0.0, 'right': 0.0},
            'front': {'left': 0.0, 'right': 0.0}
        })
        
        # Manos ocultas DETRÁS del cuerpo
        left_behind = (depth_analysis['left_hand']['behind'] and 
                    not depth_analysis['left_hand']['front'] and  # Exclusivamente detrás
                    depth_analysis['left_hand']['depth_confidence'] > 0.6 and
                    consistency['behind']['left'] > self.depth_consistency_threshold)
        
        right_behind = (depth_analysis['right_hand']['behind'] and 
                      not depth_analysis['right_hand']['front'] and  # Exclusivamente detrás
                      depth_analysis['right_hand']['depth_confidence'] > 0.6 and
                      consistency['behind']['right'] > self.depth_consistency_threshold)
        
        return left_behind or right_behind

    def detect_hand_under_clothes_3d(self, pose_landmarks, hand_landmarks):
        """Detección de manos en posición de bolsillos delanteros"""
        if pose_landmarks is None or hand_landmarks is None or len(hand_landmarks) == 0:
            self.person_data['hand_under_clothes_frames'] = 0
            return False

        try:
            # Landmarks clave para el torso (con verificación None)
            left_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
            right_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
            left_hip = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
            right_hip = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
            
            # Verificar si alguno es None (forma correcta para arrays numpy)
            if any(x is None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return False
                
            # Puntos de referencia para la zona de bolsillos delanteros
            torso_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            belly_point = torso_center + (hip_center - torso_center) * 0.5  # 30% hacia abajo
            
            # Área de detección
            pocket_zone_width = abs(left_shoulder[0] - right_shoulder[0]) * 0.3
            pocket_zone_height = abs(torso_center[1] - hip_center[1]) * 0.5
            
        except (AttributeError, KeyError):
            return False

        detection_found = False
        
        for hand_lm in hand_landmarks:
            hand_center = np.mean([self.get_3d_position(lm) for lm in hand_lm.landmark], axis=0)
            
            # 1. Verificar posición horizontal
            within_width = abs(hand_center[0] - belly_point[0]) < pocket_zone_width
            
            # 2. Verificar posición vertical
            within_height = (hip_center[1] > hand_center[1] > belly_point[1])
            
            # 3. Verificar profundidad
            torso_depth = np.mean([left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]])
            in_front = hand_center[2] < torso_depth - 0.03
            
            # 4. Verificar ángulo del codo
            elbow = None
            if hand_center[0] < belly_point[0]:  # Mano izquierda
                elbow_pos = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            else:  # Mano derecha
                elbow_pos = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                
            elbow = self.get_3d_position(elbow_pos) if elbow_pos else None
            good_angle = True  # Por defecto en caso de no tener codo
            
            if elbow is not None:
                shoulder_ref = left_shoulder if hand_center[0] < belly_point[0] else right_shoulder
                v1 = elbow - shoulder_ref
                v2 = hand_center - elbow
                angle = self.calculate_3d_angle(v1, v2)
                good_angle = 60 < angle < 120

            # print("----------------------------------------------------------------------")
            # print("within_width: " + str(within_width) + "\nwithin_height:" + str(within_height) + "\nin_front: " + str(in_front) + "\ngood_angle: " + str(good_angle))
            if within_width and within_height and good_angle:
                detection_found = True
                break

        if detection_found:
                return True
        
        return False

    def detect_excessive_gaze_3d(self, face_landmarks, pose_landmarks, frame_shape, current_time):
        if face_landmarks is None and pose_landmarks is None:
            return False

        gaze_vector_3d = None

        if face_landmarks:
            left_eye_center_3d = np.mean([self.get_3d_position(face_landmarks.landmark[idx]) for idx in self.LEFT_EYE_INDICES], axis=0)
            right_eye_center_3d = np.mean([self.get_3d_position(face_landmarks.landmark[idx]) for idx in self.RIGHT_EYE_INDICES], axis=0)
            eyes_center_3d = (left_eye_center_3d + right_eye_center_3d) / 2

            has_iris_data = len(face_landmarks.landmark) > 468
            if has_iris_data:
                left_iris_3d = np.mean([self.get_3d_position(face_landmarks.landmark[idx]) for idx in self.IRIS_INDICES[:3]], axis=0)
                right_iris_3d = np.mean([self.get_3d_position(face_landmarks.landmark[idx]) for idx in self.IRIS_INDICES[3:]], axis=0)
                iris_center_3d = (left_iris_3d + right_iris_3d) / 2
                gaze_vector_3d = iris_center_3d - eyes_center_3d
            else:
                nose_tip_3d = self.get_3d_position(face_landmarks.landmark[4])
                gaze_vector_3d = nose_tip_3d - eyes_center_3d

        elif pose_landmarks:
            nose_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE])
            left_eye_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE])
            right_eye_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE])
            eyes_center_3d = (left_eye_3d + right_eye_3d) / 2
            gaze_vector_3d = nose_3d - eyes_center_3d

        if gaze_vector_3d is None:
            return False

        gaze_magnitude = np.linalg.norm(gaze_vector_3d)
        if gaze_magnitude > 0:
            gaze_vector_3d = gaze_vector_3d / gaze_magnitude
        else:
            return False

        self.person_data['gaze_directions_3d'].append(gaze_vector_3d)

        if len(self.person_data['gaze_directions_3d']) < 2:
            return False

        prev_gaze_3d = self.person_data['gaze_directions_3d'][-2]
        angle_change_3d = self.calculate_3d_angle(gaze_vector_3d, prev_gaze_3d)

        if angle_change_3d > np.degrees(self.gaze_angle_threshold_3d):
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

    def reset_hidden_hands_counters(self):
        """Reinicia los contadores de manos ocultas"""
        self.person_data['hidden_hands_frames'] = 0
        self.person_data['hidden_hands_duration'] = 0
        if 'hidden_hands' in self.person_data['suspicious_start_times']:
            del self.person_data['suspicious_start_times']['hidden_hands']
        if 'hidden_hands' in self.person_data['alerted']:
            self.person_data['alerted'].remove('hidden_hands')

    def reset_hand_under_clothes_counters(self):
        """Reinicia los contadores de manos bajo ropa"""
        self.person_data['hand_under_clothes_frames'] = 0
        self.person_data['hand_under_clothes_duration'] = 0
        if 'hand_under_clothes' in self.person_data['suspicious_start_times']:
            del self.person_data['suspicious_start_times']['hand_under_clothes']
        if 'hand_under_clothes' in self.person_data['alerted']:
            self.person_data['alerted'].remove('hand_under_clothes')

    def draw_landmarks_optimized(self, frame, pose_results, hand_results, face_results):
        """Dibuja landmarks optimizados"""
        # Dibujar pose
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Dibujar manos
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # Dibujar cara optimizado
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
                )

    def process_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(input_path) if not self.with_camera else cv2.VideoCapture(self.connection if self.connection else 0)
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el video")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if not self.with_camera:    
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fourcc = cv2.VideoWriter_fourcc(*'VP90')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_idx = 0
            frame_counter = 0
            detections = []

            if self.frame_skip > 0:
                self.hidden_hands_frame_threshold = max(1, self.hidden_hands_frame_threshold // self.frame_skip)
                self.hand_under_clothes_frame_threshold = max(1, self.hand_under_clothes_frame_threshold // self.frame_skip)

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
                    
                    # Procesar frame
                    pose_results = self.pose.process(frame_rgb)
                    hand_results = None
                    face_results = None

                    if pose_results and pose_results.pose_landmarks:
                        hand_results = self.hands.process(frame_rgb)
                        face_results = self.face_mesh.process(frame_rgb)

                        current_time = frame_idx / fps
                        behaviors = {
                            'hidden_hands': False,
                            'excessive_gaze': False,
                            'hand_under_clothes': False
                        }

                        torso_landmarks_3d = [
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
                        ]
                        position_3d = np.mean([pos for pos in torso_landmarks_3d if pos is not None], axis=0)

                        # Análisis de manos
                        multi_hand_landmarks = hand_results.multi_hand_landmarks if hand_results and hand_results.multi_hand_landmarks else []
                        depth_analysis = self.analyze_hand_depth_position(pose_results.pose_landmarks, multi_hand_landmarks)
                        self._last_depth_analysis = depth_analysis

                        # Detección de comportamientos
                        hands_in_pockets_3d = self.detect_hand_pockets_3d(pose_results.pose_landmarks, multi_hand_landmarks, frame.shape)
                        if hands_in_pockets_3d:
                            self.person_data['hidden_hands_frames'] += self.frame_skip
                            self.person_data['hidden_hands_duration'] = self.person_data['hidden_hands_frames'] / fps

                            if 'hidden_hands' not in self.person_data['suspicious_start_times']:
                                self.person_data['suspicious_start_times']['hidden_hands'] = current_time
                                self.person_data['hidden_hands_position_3d'] = position_3d

                            if (self.person_data['hidden_hands_frames'] > self.hidden_hands_frame_threshold and
                                    self.person_data['hidden_hands_duration'] >= self.hidden_hands_time_threshold):
                                behaviors['hidden_hands'] = True
                                if 'hidden_hands' not in self.person_data['alerted']:
                                    self.person_data['alerted'].add('hidden_hands')
                        else:
                            self.reset_hidden_hands_counters()

                        # Detección de mirada
                        face_landmarks = face_results.multi_face_landmarks[0] if face_results and face_results.multi_face_landmarks else None
                        if face_landmarks:
                            excessive_gaze_3d = self.detect_excessive_gaze_3d(face_landmarks, pose_results.pose_landmarks, frame.shape, current_time)
                            if excessive_gaze_3d:
                                behaviors['excessive_gaze'] = True
                                if 'excessive_gaze' not in self.person_data['alerted']:
                                    self.person_data['alerted'].add('excessive_gaze')

                        # Detección manos bajo ropa
                        hand_under_clothes_3d = self.detect_hand_under_clothes_3d(
                            pose_results.pose_landmarks, 
                            multi_hand_landmarks
                        )
                        
                        if hand_under_clothes_3d:
                            self.person_data['hand_under_clothes_frames'] += self.frame_skip
                            self.person_data['hand_under_clothes_duration'] = self.person_data['hand_under_clothes_frames'] / fps

                            if 'hand_under_clothes' not in self.person_data['suspicious_start_times']:
                                self.person_data['suspicious_start_times']['hand_under_clothes'] = current_time

                            if (self.person_data['hand_under_clothes_frames'] > self.hand_under_clothes_frame_threshold and
                                    self.person_data['hand_under_clothes_duration'] >= self.hand_under_clothes_time_threshold):
                                behaviors['hand_under_clothes'] = True
                                if 'hand_under_clothes' not in self.person_data['alerted']:
                                    self.person_data['alerted'].add('hand_under_clothes')
                        else:
                            self.reset_hand_under_clothes_counters()

                        # Registrar detecciones
                        if any(behaviors.values()):
                            detections.append({
                                'timestamp': current_time,
                                'behaviors': [b for b, detected in behaviors.items() if detected]
                            })

                        # Dibujar landmarks
                        self.draw_landmarks_optimized(output_frame, pose_results, hand_results, face_results)

                    num_people = 1 if pose_results and pose_results.pose_landmarks else 0
                    cv2.putText(output_frame, f"Personas: {num_people}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Codificar y enviar frame
                _, buffer = cv2.imencode('.jpg', output_frame)
                jpg_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data = {
                    'frame': jpg_b64,
                    'progress': f"{round(frame_idx / total_frames * 100)}" if not self.with_camera else None,
                    'detections': list(detections[-1]['behaviors']) if detections else None
                }

                if not self.with_camera:
                    out.write(output_frame)

                if process_this_frame:
                    yield frame_data
            
            yield detections
            
        finally:
            cap.release()
            if not self.with_camera and 'out' in locals():
                out.release()