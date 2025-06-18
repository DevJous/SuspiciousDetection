from multiprocessing import process
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import defaultdict, deque
import base64

class BehaviorDetector:
    def __init__(self, frame_skip=3, connection = None, with_camera=False):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.frame_skip = frame_skip
        self.with_camera = with_camera
        self.connection = connection

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

        self.person_data = {
            'positions': deque(maxlen=30),
            'hidden_hands_frames': 0,
            'hidden_hands_duration': 0,
            'hidden_hands_position': None,
            'suspicious_start_times': {},
            'alerted': set(),
            'gaze_directions': deque(maxlen=60),
            'gaze_change_counter': 0,
            'last_significant_gaze_time': 0,
            'hand_under_clothes_frames': 0,
            'hand_under_clothes_duration': 0,
        }

        self.hidden_hands_frame_threshold = 25
        self.hidden_hands_time_threshold = 3.0
        self.gaze_angle_threshold = 0.3
        self.gaze_changes_threshold = 8
        self.gaze_time_window = 5.0
        self.hand_under_clothes_frame_threshold = 20
        self.hand_under_clothes_time_threshold = 2.0
        self.arm_angle_threshold_min = 90
        self.arm_angle_threshold_max = 140
        self.confidence_threshold = 0.7

        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 374, 373, 390]
        self.IRIS_INDICES = [468, 469, 470, 471, 472, 473]

    def detect_hand_pockets(self, pose_landmarks, hand_landmarks, frame_shape):
        if not pose_landmarks:
            return False

        h, w, _ = frame_shape

        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2

        visible_hands = set()
        if hand_landmarks:
            for hand_lm in hand_landmarks:
                hand_x = sum(lm.x for lm in hand_lm.landmark) / len(hand_lm.landmark)

                if hand_x < shoulder_mid_x:
                    visible_hands.add('left')
                else:
                    visible_hands.add('right')

        hands_in_pockets = False

        if 'left' not in visible_hands:
            hip_wrist_distance_left = math.sqrt((left_hip.x - left_wrist.x) ** 2 + (left_hip.y - left_wrist.y) ** 2)
            behind_back_left = left_wrist.z > left_hip.z + 0.1
            if hip_wrist_distance_left < 0.15 or behind_back_left:
                hands_in_pockets = True

        if 'right' not in visible_hands:
            hip_wrist_distance_right = math.sqrt(
                (right_hip.x - right_wrist.x) ** 2 + (right_hip.y - right_wrist.y) ** 2)
            behind_back_right = right_wrist.z > right_hip.z + 0.1
            if hip_wrist_distance_right < 0.15 or behind_back_right:
                hands_in_pockets = True

        return hands_in_pockets

    def detect_hand_under_clothes(self, pose_landmarks):
        if not pose_landmarks:
            return False

        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

        def calculate_arm_angle(shoulder, elbow, wrist):
            v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
            v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
            
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms == 0:
                return 0
                
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg

        def is_hand_near_pocket(wrist, hip, threshold=0.25):
            distance = math.sqrt((wrist.x - hip.x) ** 2 + (wrist.y - hip.y) ** 2)
            return distance < threshold

        left_angle = calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_arm_angle(right_shoulder, right_elbow, right_wrist)

        left_near_pocket = is_hand_near_pocket(left_wrist, left_hip)
        right_near_pocket = is_hand_near_pocket(right_wrist, right_hip)

        suspicious_left = (self.arm_angle_threshold_min <= left_angle <= self.arm_angle_threshold_max) and left_near_pocket
        suspicious_right = (self.arm_angle_threshold_min <= right_angle <= self.arm_angle_threshold_max) and right_near_pocket

        return suspicious_left or suspicious_right

    def detect_excessive_gaze(self, face_landmarks, pose_landmarks, frame_shape, current_time):
        if not face_landmarks and not pose_landmarks:
            return False

        if face_landmarks:
            left_eye_center = np.mean([[face_landmarks.landmark[idx].x,
                                        face_landmarks.landmark[idx].y]
                                       for idx in self.LEFT_EYE_INDICES], axis=0)

            right_eye_center = np.mean([[face_landmarks.landmark[idx].x,
                                         face_landmarks.landmark[idx].y]
                                        for idx in self.RIGHT_EYE_INDICES], axis=0)

            eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)

            has_iris_data = len(face_landmarks.landmark) > 468

            if has_iris_data:
                left_iris = np.mean([[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
                                     for idx in self.IRIS_INDICES[:3]], axis=0)
                right_iris = np.mean([[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
                                      for idx in self.IRIS_INDICES[3:]], axis=0)

                gaze_vector = np.mean([
                    left_iris - left_eye_center,
                    right_iris - right_eye_center
                ], axis=0)
            else:
                nose_tip = np.array([face_landmarks.landmark[4].x, face_landmarks.landmark[4].y])
                gaze_vector = nose_tip - eyes_center

        elif pose_landmarks:
            nose = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
                             pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y])

            left_eye = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x,
                                 pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y])

            right_eye = np.array([pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x,
                                  pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y])

            eyes_center = np.mean([left_eye, right_eye], axis=0)
            gaze_vector = nose - eyes_center

        gaze_magnitude = np.linalg.norm(gaze_vector)
        if gaze_magnitude > 0:
            gaze_vector = gaze_vector / gaze_magnitude
        else:
            return False

        self.person_data['gaze_directions'].append(gaze_vector)

        if len(self.person_data['gaze_directions']) < 2:
            return False

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

    def process_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(input_path) if not self.with_camera else cv2.VideoCapture(self.connection if self.connection else 0)
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

                    pose_results = self.pose.process(frame_rgb)
                    
                    if pose_results.pose_landmarks:

                        hand_results = self.hands.process(frame_rgb)
                        face_results = self.face_mesh.process(frame_rgb)

                        current_time = frame_idx / fps

                        behaviors = {
                            'hidden_hands': False,
                            'excessive_gaze': False,
                            'hand_under_clothes': False
                        }

                        if pose_results.pose_landmarks:
                            torso_landmarks = [
                                pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                                pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                                pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
                            ]

                            current_x = sum(l.x for l in torso_landmarks) / len(torso_landmarks)
                            current_y = sum(l.y for l in torso_landmarks) / len(torso_landmarks)
                            position = (current_x, current_y)

                            multi_hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
                            hands_in_pockets = self.detect_hand_pockets(pose_results.pose_landmarks, multi_hand_landmarks,
                                                                        frame.shape)

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

                            hand_under_clothes = self.detect_hand_under_clothes(pose_results.pose_landmarks)

                            if hand_under_clothes:
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
                                    pos_x, pos_y = int(position[0] * w), int(position[1] * h - 90)
                                    cv2.putText(output_frame, "Mano bajo ropa",
                                            (pos_x - 60, pos_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                self.person_data['hand_under_clothes_frames'] = 0
                                self.person_data['hand_under_clothes_duration'] = 0
                                if 'hand_under_clothes' in self.person_data['suspicious_start_times']:
                                    del self.person_data['suspicious_start_times']['hand_under_clothes']
                                if 'hand_under_clothes' in self.person_data['alerted']:
                                    self.person_data['alerted'].remove('hand_under_clothes')

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
                    else:
                        if self.with_camera:
                            cap.release()

                    num_people = 1 if pose_results.pose_landmarks else 0
                    cv2.putText(output_frame, f"Personas: {num_people}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
                _, buffer = cv2.imencode('.jpg', output_frame)
                jpg_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data = {
                    'frame': jpg_b64,
                    'progress': f"{round(frame_idx / total_frames * 100)}",
                    'detections': list(detections[-1]['behaviors']) if detections else None
                } if not self.with_camera else {
                    'frame': jpg_b64,
                    'detections': list(detections[-1]['behaviors']) if detections else None
                }

                if not self.with_camera:
                    out.write(output_frame)

                yield frame_data

            yield detections
        finally:
            cap.release()

            if not self.with_camera:
                out.release()