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
        }

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
        self.depth_threshold = 0.15
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

    def detect_hand_pockets_3d(self, pose_landmarks, hand_landmarks, frame_shape):
        if not pose_landmarks:
            return False

        left_hip_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        left_wrist_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST])
        right_wrist_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        left_shoulder_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder_3d = self.get_3d_position(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])

        shoulder_mid_3d = (left_shoulder_3d + right_shoulder_3d) / 2

        visible_hands = set()
        if hand_landmarks:
            for hand_lm in hand_landmarks:
                hand_center_3d = np.mean([self.get_3d_position(lm) for lm in hand_lm.landmark], axis=0)
                if hand_center_3d[0] < shoulder_mid_3d[0]:
                    visible_hands.add('left')
                else:
                    visible_hands.add('right')

        hands_in_pockets = False

        if 'left' not in visible_hands:
            hip_wrist_distance_3d = self.calculate_3d_distance(left_hip_3d, left_wrist_3d)
            behind_back_3d = left_wrist_3d[2] > left_hip_3d[2] + self.depth_threshold
            depth_hidden = abs(left_wrist_3d[2] - left_hip_3d[2]) > self.depth_threshold
            if hip_wrist_distance_3d < 0.15 or behind_back_3d or depth_hidden:
                hands_in_pockets = True

        if 'right' not in visible_hands:
            hip_wrist_distance_3d = self.calculate_3d_distance(right_hip_3d, right_wrist_3d)
            behind_back_3d = right_wrist_3d[2] > right_hip_3d[2] + self.depth_threshold
            depth_hidden = abs(right_wrist_3d[2] - right_hip_3d[2]) > self.depth_threshold
            if hip_wrist_distance_3d < 0.15 or behind_back_3d or depth_hidden:
                hands_in_pockets = True

        return hands_in_pockets

    def detect_hand_under_clothes_3d(self, pose_landmarks):
        if not pose_landmarks:
            return False

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

    def analyze_movement_3d(self, position_3d, current_time):
        self.person_data['positions_3d'].append((position_3d, current_time))

        if len(self.person_data['positions_3d']) < 2:
            return False

        prev_pos, prev_time = self.person_data['positions_3d'][-2]
        curr_pos, curr_time = self.person_data['positions_3d'][-1]

        time_diff = curr_time - prev_time
        if time_diff > 0:
            velocity_3d = (curr_pos - prev_pos) / time_diff
            velocity_magnitude = np.linalg.norm(velocity_3d)
            self.person_data['velocity_3d'].append(velocity_3d)

            if len(self.person_data['velocity_3d']) >= 2:
                prev_vel = self.person_data['velocity_3d'][-2]
                acceleration_3d = (velocity_3d - prev_vel) / time_diff
                self.person_data['acceleration_3d'].append(acceleration_3d)

            return velocity_magnitude > self.suspicious_velocity_threshold

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
                        'hand_under_clothes': False,
                        'suspicious_movement': False
                    }

                    if pose_results.pose_landmarks:
                        torso_landmarks_3d = [
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]),
                            self.get_3d_position(pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
                        ]

                        position_3d = np.mean(torso_landmarks_3d, axis=0)
                        suspicious_movement = self.analyze_movement_3d(position_3d, current_time)

                        if suspicious_movement:
                            behaviors['suspicious_movement'] = True

                        multi_hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
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
                                cv2.putText(output_frame, "Manos ocultas 3D", (pos_x - 60, pos_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
                            cv2.putText(output_frame, "Mirada excesiva 3D", (pos_x - 60, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
                                cv2.putText(output_frame, "Mano bajo ropa 3D", (pos_x - 60, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            self.person_data['hand_under_clothes_frames'] = 0
                            self.person_data['hand_under_clothes_duration'] = 0
                            if 'hand_under_clothes' in self.person_data['suspicious_start_times']:
                                del self.person_data['suspicious_start_times']['hand_under_clothes']
                            if 'hand_under_clothes' in self.person_data['alerted']:
                                self.person_data['alerted'].remove('hand_under_clothes')

                        if suspicious_movement:
                            h, w, _ = frame.shape
                            pos_x, pos_y = int(position_3d[0] * w), int(position_3d[1] * h + 30)
                            cv2.putText(output_frame, "Movimiento sospechoso", (pos_x - 80, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                        if any(behaviors.values()):
                            detections.append({
                                'timestamp': current_time,
                                'behaviors': [b for b, detected in behaviors.items() if detected],
                                'position_3d': position_3d.tolist(),
                                'depth_info': position_3d[2]
                            })

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
                    'detections': list(detections[-1]['behaviors']) if detections else None,
                    'depth_data': detections[-1]['depth_info'] if detections else None
                }

                yield frame_data
                out.write(output_frame)

            yield detections
        finally:
            cap.release()
            out.release()