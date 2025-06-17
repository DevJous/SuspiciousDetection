from multiprocessing import process
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import defaultdict, deque
import base64

class BehaviorDetector3D:
    def __init__(self, frame_skip=3):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.frame_skip = frame_skip

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
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
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
            # Nuevos datos para análisis de profundidad
            'hand_depth_history': deque(maxlen=15),
            'torso_depth_reference': deque(maxlen=10),
            'body_orientation': deque(maxlen=5),
        }

        # Parámetros mejorados para detección de profundidad
        self.hidden_hands_frame_threshold = 25
        self.hidden_hands_time_threshold = 3.0
        self.gaze_angle_threshold_3d = 0.4
        self.gaze_changes_threshold = 8
        self.gaze_time_window = 5.0
        self.hand_under_clothes_frame_threshold = 20
        self.hand_under_clothes_time_threshold = 2.0
        self.arm_angle_threshold_min = 90
        self.arm_angle_threshold_max = 140
        self.confidence_threshold = 0.7
        
        # Umbrales de profundidad más precisos
        self.depth_threshold_behind = 0.08  # Manos detrás del cuerpo
        self.depth_threshold_front = 0.06   # Manos delante del cuerpo
        self.depth_consistency_threshold = 0.7  # Consistencia en frames
        self.torso_depth_variance_threshold = 0.03  # Varianza permitida en profundidad del torso
        
        self.movement_threshold = 0.02
        self.suspicious_velocity_threshold = 1.5

        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 374, 373, 390]
        self.IRIS_INDICES = [468, 469, 470, 471, 472, 473]

    def get_3d_position(self, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    def calculate_3d_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def calculate_3d_angle(self, v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_body_orientation(self, pose_landmarks):
        """Calcula la orientación del cuerpo en el espacio 3D"""
        left_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        left_hip = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        
        # Vector del torso (hombros a caderas)
        torso_vector = ((left_hip + right_hip) / 2) - ((left_shoulder + right_shoulder) / 2)
        
        # Vector de los hombros (izquierdo a derecho)
        shoulder_vector = right_shoulder - left_shoulder
        
        # Normal del plano del cuerpo
        body_normal = np.cross(torso_vector, shoulder_vector)
        body_normal = body_normal / np.linalg.norm(body_normal)
        
        return body_normal, torso_vector, shoulder_vector

    def get_torso_depth_reference(self, pose_landmarks):
        """Obtiene la profundidad de referencia del torso"""
        chest_landmarks = [
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]),
            self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        ]
        
        # Profundidad promedio del torso
        torso_depth = np.mean([pos[2] for pos in chest_landmarks])
        
        # Guardar en historial para estabilidad
        self.person_data['torso_depth_reference'].append(torso_depth)
        
        return torso_depth

    def analyze_hand_depth_position(self, pose_landmarks, hand_landmarks):
        """Analiza la posición de profundidad de las manos con mayor precisión"""
        if not pose_landmarks:
            return {
                'left_hand': {'behind': False, 'front': False, 'depth_confidence': 0},
                'right_hand': {'behind': False, 'front': False, 'depth_confidence': 0}
            }

        # Obtener referencias del cuerpo
        torso_depth = self.get_torso_depth_reference(pose_landmarks)
        body_normal, torso_vector, shoulder_vector = self.calculate_body_orientation(pose_landmarks)
        
        # Posiciones de muñecas y hombros
        left_wrist = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST])
        right_wrist = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        left_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        
        # Análisis de manos visibles vs detectadas por pose
        visible_hands = {'left': False, 'right': False}
        hand_depths = {'left': None, 'right': None}
        
        # Identificar manos visibles con MediaPipe Hands
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
            """Analiza una mano individual"""
            result = {'behind': False, 'front': False, 'depth_confidence': 0}
            
            # Diferencia de profundidad respecto al torso
            wrist_depth_diff = wrist_pos[2] - torso_depth
            shoulder_depth_diff = shoulder_pos[2] - torso_depth
            
            # Si la mano no es visible pero la muñeca está detectada
            if not is_visible:
                # Mano detrás del cuerpo
                if wrist_depth_diff > self.depth_threshold_behind:
                    result['behind'] = True
                    result['depth_confidence'] = min(1.0, abs(wrist_depth_diff) / self.depth_threshold_behind)
                
                # Verificación adicional: distancia a la cadera para bolsillos
                if hand_side == 'left':
                    hip_pos = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
                else:
                    hip_pos = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
                
                hip_distance = self.calculate_3d_distance(wrist_pos, hip_pos)
                if hip_distance < 0.15 and wrist_depth_diff > 0.03:
                    result['behind'] = True
                    result['depth_confidence'] = max(result['depth_confidence'], 0.8)
            
            else:
                # Mano visible - verificar si está sospechosamente adelante
                if hand_depth and (hand_depth - torso_depth) < -self.depth_threshold_front:
                    result['front'] = True
                    result['depth_confidence'] = min(1.0, abs(hand_depth - torso_depth) / self.depth_threshold_front)
                
                # Verificación de proximidad al torso para "mano bajo ropa"
                torso_center = np.mean([left_shoulder, right_shoulder], axis=0)
                hand_pos = wrist_pos if not hand_depth else np.array([wrist_pos[0], wrist_pos[1], hand_depth])
                
                torso_distance = self.calculate_3d_distance(hand_pos[:2], torso_center[:2])  # Solo X,Y
                if torso_distance < 0.2 and abs(hand_pos[2] - torso_depth) < 0.1:
                    result['front'] = True
                    result['depth_confidence'] = max(result['depth_confidence'], 0.7)
            
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
        """Detección mejorada de manos en bolsillos usando análisis de profundidad"""
        if not pose_landmarks:
            return False

        depth_analysis = self.analyze_hand_depth_position(pose_landmarks, hand_landmarks)
        consistency = depth_analysis.get('consistency', {
            'behind': {'left': 0.0, 'right': 0.0},
            'front': {'left': 0.0, 'right': 0.0}
        })
        
        # Verificar que consistency['behind'] es un diccionario
        if not isinstance(consistency['behind'], dict):
            return False
        
        # Manos ocultas detrás del cuerpo
        left_behind = (depth_analysis['left_hand']['behind'] and 
                    depth_analysis['left_hand']['depth_confidence'] > 0.6 and
                    consistency['behind']['left'] > self.depth_consistency_threshold)
        
        right_behind = (depth_analysis['right_hand']['behind'] and 
                    depth_analysis['right_hand']['depth_confidence'] > 0.6 and
                    consistency['behind']['right'] > self.depth_consistency_threshold)
        
        return left_behind or right_behind

    def detect_hand_under_clothes_3d(self, pose_landmarks):
        """Detección mejorada de manos bajo ropa usando análisis de profundidad y ángulos de brazo"""
        if not pose_landmarks:
            return False

        # Obtener posiciones 3D de los landmarks clave
        left_shoulder_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        left_elbow_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        left_wrist_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST])
        
        right_shoulder_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        right_elbow_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        right_wrist_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])

        # Obtener referencia del torso
        torso_depth = self.get_torso_depth_reference(pose_landmarks)
        torso_center_3d = np.mean([left_shoulder_3d, right_shoulder_3d], axis=0)
        
        def analyze_arm_position(shoulder_3d, elbow_3d, wrist_3d, torso_center, torso_depth):
            """Analiza la posición del brazo para detectar mano bajo ropa"""
            
            # 1. Calcular ángulo del brazo (hombro-codo-muñeca)
            v1 = shoulder_3d - elbow_3d  # Vector hombro a codo
            v2 = wrist_3d - elbow_3d     # Vector codo a muñeca
            arm_angle = self.calculate_3d_angle(v1, v2)
            
            # 2. Verificar si el ángulo está en el rango sospechoso (120-140 grados)
            angle_suspicious = 120 <= arm_angle <= 140
            
            if not angle_suspicious:
                return False, 0.0, {}
            
            # 3. Analizar profundidad - mano delante del cuerpo
            wrist_depth_diff = torso_depth - wrist_3d[2]  # Diferencia de profundidad
            hand_in_front = wrist_depth_diff > 0.03  # Mano está delante del torso
            
            # 4. Verificar proximidad al torso en el plano X-Y
            horizontal_distance = np.linalg.norm([wrist_3d[0] - torso_center[0], wrist_3d[1] - torso_center[1]])
            close_to_torso = horizontal_distance < 0.25  # Cerca del centro del torso
            
            # 5. Verificar posición vertical (altura similar al torso)
            vertical_diff = abs(wrist_3d[1] - torso_center[1])
            appropriate_height = vertical_diff < 0.3  # Altura apropiada para tocar el torso
            
            # 6. Calcular vector de dirección del antebrazo
            forearm_vector = wrist_3d - elbow_3d
            forearm_magnitude = np.linalg.norm(forearm_vector)
            
            if forearm_magnitude > 0:
                forearm_direction = forearm_vector / forearm_magnitude
                
                # Vector hacia el centro del torso desde el codo
                to_torso_vector = torso_center - elbow_3d
                to_torso_magnitude = np.linalg.norm(to_torso_vector)
                
                if to_torso_magnitude > 0:
                    to_torso_direction = to_torso_vector / to_torso_magnitude
                    
                    # Calcular similitud de dirección (producto punto)
                    direction_similarity = np.dot(forearm_direction, to_torso_direction)
                    pointing_to_torso = direction_similarity > 0.3  # Antebrazo apunta hacia el torso
                else:
                    pointing_to_torso = False
            else:
                pointing_to_torso = False
            
            # 7. Calcular confianza basada en múltiples factores
            confidence = 0.0
            
            if angle_suspicious:
                confidence += 0.3
            if hand_in_front:
                confidence += 0.25
            if close_to_torso:
                confidence += 0.2
            if appropriate_height:
                confidence += 0.15
            if pointing_to_torso:
                confidence += 0.1
            
            # Bonus por profundidad específica
            if 0.03 < wrist_depth_diff < 0.15:
                confidence += 0.1
            
            # 8. Determinar si es sospechoso
            is_suspicious = (angle_suspicious and hand_in_front and 
                            close_to_torso and confidence > 0.7)
            
            # Información de depuración
            debug_info = {
                'arm_angle': arm_angle,
                'wrist_depth_diff': wrist_depth_diff,
                'horizontal_distance': horizontal_distance,
                'vertical_diff': vertical_diff,
                'hand_in_front': hand_in_front,
                'close_to_torso': close_to_torso,
                'appropriate_height': appropriate_height,
                'pointing_to_torso': pointing_to_torso,
                'direction_similarity': direction_similarity if 'direction_similarity' in locals() else 0.0
            }
            
            return is_suspicious, confidence, debug_info
        
        # Analizar brazo izquierdo
        left_suspicious, left_confidence, left_debug = analyze_arm_position(
            left_shoulder_3d, left_elbow_3d, left_wrist_3d, torso_center_3d, torso_depth
        )
        
        # Analizar brazo derecho
        right_suspicious, right_confidence, right_debug = analyze_arm_position(
            right_shoulder_3d, right_elbow_3d, right_wrist_3d, torso_center_3d, torso_depth
        )
        
        # Guardar información de depuración para visualización
        self._arm_analysis_debug = {
            'left': left_debug,
            'right': right_debug,
            'left_confidence': left_confidence,
            'right_confidence': right_confidence,
            'torso_depth': torso_depth
        }
        
        # Retornar True si cualquier brazo está en posición sospechosa
        return left_suspicious or right_suspicious

    def _detect_hand_under_clothes_original(self, pose_landmarks):
        """Método original de detección de manos bajo ropa como fallback"""
        left_shoulder_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        left_elbow_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        left_wrist_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST])
        
        right_shoulder_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        right_elbow_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        right_wrist_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])

        left_hip_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])

        def calculate_arm_angle_3d(shoulder_3d, elbow_3d, wrist_3d):
            v1 = shoulder_3d - elbow_3d
            v2 = wrist_3d - elbow_3d
            return self.calculate_3d_angle(v1, v2)

        def is_hand_near_pocket_3d(wrist_3d, hip_3d, threshold=0.25):
            return self.calculate_3d_distance(wrist_3d, hip_3d) < threshold

        def is_hand_depth_suspicious(wrist_3d, shoulder_3d, threshold=0.1):
            return abs(wrist_3d[2] - shoulder_3d[2]) > threshold

        left_angle_3d = calculate_arm_angle_3d(left_shoulder_3d, left_elbow_3d, left_wrist_3d)
        right_angle_3d = calculate_arm_angle_3d(right_shoulder_3d, right_elbow_3d, right_wrist_3d)

        left_near_pocket = is_hand_near_pocket_3d(left_wrist_3d, left_hip_3d)
        right_near_pocket = is_hand_near_pocket_3d(right_wrist_3d, right_hip_3d)

        left_depth_suspicious = is_hand_depth_suspicious(left_wrist_3d, left_shoulder_3d)
        right_depth_suspicious = is_hand_depth_suspicious(right_wrist_3d, right_shoulder_3d)

        suspicious_left = (self.arm_angle_threshold_min <= left_angle_3d <= self.arm_angle_threshold_max) and (left_near_pocket or left_depth_suspicious)
        suspicious_right = (self.arm_angle_threshold_min <= right_angle_3d <= self.arm_angle_threshold_max) and (right_near_pocket or right_depth_suspicious)

        return suspicious_left or suspicious_right

    def detect_excessive_gaze_3d(self, face_landmarks, pose_landmarks, frame_shape, current_time):
        if not face_landmarks and not pose_landmarks:
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

                    pose_results = self.pose.process(frame_rgb)
                    hand_results = self.hands.process(frame_rgb)
                    face_results = self.face_mesh.process(frame_rgb)

                    current_time = frame_idx / fps

                    behaviors = {
                        'hidden_hands': False,
                        'excessive_gaze': False,
                        'hand_under_clothes': False
                    }

                    if pose_results.pose_landmarks:
                        torso_landmarks_3d = [
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
                        ]

                        position_3d = np.mean(torso_landmarks_3d, axis=0)

                        # Realizar análisis de profundidad de manos
                        multi_hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
                        depth_analysis = self.analyze_hand_depth_position(pose_results.pose_landmarks, multi_hand_landmarks)
                        self._last_depth_analysis = depth_analysis  # Guardar para uso en otras funciones
                        
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

                                h, w, _ = frame.shape
                                pos_x, pos_y = int(position_3d[0] * w), int(position_3d[1] * h)
                                
                                # Mostrar información de profundidad
                                depth_info = f"Profundidad: {depth_analysis['torso_depth']:.3f}"
                                cv2.putText(output_frame, "Manos ocultas (Detras)", (pos_x - 60, pos_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(output_frame, depth_info, (pos_x - 60, pos_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        else:
                            self.person_data['hidden_hands_frames'] = 0
                            self.person_data['hidden_hands_duration'] = 0
                            if 'hidden_hands' in self.person_data['suspicious_start_times']:
                                del self.person_data['suspicious_start_times']['hidden_hands']
                            if 'hidden_hands' in self.person_data['alerted']:
                                self.person_data['alerted'].remove('hidden_hands')

                        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
                        excessive_gaze_3d = self.detect_excessive_gaze_3d(face_landmarks, pose_results.pose_landmarks, frame.shape, current_time)

                        if excessive_gaze_3d:
                            behaviors['excessive_gaze'] = True

                            if 'excessive_gaze' not in self.person_data['alerted']:
                                self.person_data['alerted'].add('excessive_gaze')

                            h, w, _ = frame.shape
                            pos_x, pos_y = int(position_3d[0] * w), int(position_3d[1] * h - 60)

                        hand_under_clothes_3d = self.detect_hand_under_clothes_3d(pose_results.pose_landmarks)

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

                                h, w, _ = frame.shape
                                pos_x, pos_y = int(position_3d[0] * w), int(position_3d[1] * h - 90)
                        else:
                            # Reset counters si no se detecta
                            self.person_data['hand_under_clothes_frames'] = 0
                            self.person_data['hand_under_clothes_duration'] = 0
                            if 'hand_under_clothes' in self.person_data['suspicious_start_times']:
                                del self.person_data['suspicious_start_times']['hand_under_clothes']
                            if 'hand_under_clothes' in self.person_data['alerted']:
                                self.person_data['alerted'].remove('hand_under_clothes')

                        if any(behaviors.values()):
                            detection_data = {
                                'timestamp': current_time,
                                'behaviors': [b for b, detected in behaviors.items() if detected],
                                'position_3d': position_3d.tolist(),
                                'depth_info': depth_analysis['torso_depth'] if depth_analysis else position_3d[2],
                                'hand_depth_analysis': {
                                    'left_hand': depth_analysis['left_hand'] if depth_analysis else None,
                                    'right_hand': depth_analysis['right_hand'] if depth_analysis else None,
                                    'consistency': depth_analysis['consistency'] if depth_analysis else None
                                }
                            }
                            detections.append(detection_data)

                        # Dibujar landmarks
                        self.mp_drawing.draw_landmarks(output_frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(output_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        if face_results.multi_face_landmarks:
                            for face_landmarks in face_results.multi_face_landmarks:
                                self.mp_drawing.draw_landmarks(output_frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)

                    num_people = 1 if pose_results.pose_landmarks else 0
                    cv2.putText(output_frame, f"Personas: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', output_frame)
                jpg_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data = {
                    'frame': jpg_b64,
                    'progress': f"{round(frame_idx / total_frames * 100)}",
                    'detections': list(detections[-1]['behaviors']) if detections else None
                }

                out.write(output_frame)
                yield frame_data
            
            yield detections
            
        finally:
            cap.release()
            out.release()