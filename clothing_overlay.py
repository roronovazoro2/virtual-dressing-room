import cv2
import mediapipe as mp
import numpy as np
from flask import jsonify
import os
import base64
from threading import Lock

class ClothingOverlay:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.size_scales = {'S': 0.9, 'M': 1.0, 'L': 1.1}
        self.current_outfit = None
        self.outfits = self.load_default_outfits()
        self.camera = None
        self.camera_lock = Lock()
        self._cache = {}

    def load_default_outfits(self):
        # Ensure outfits directory exists
        os.makedirs('static/outfits', exist_ok=True)
        
        # Create default outfit if not exists
        self.create_default_outfit()
        
        outfits = {}
        outfit_types = ['tshirt', 'jacket', 'pants']
        for outfit in outfit_types:
            path = f'static/outfits/{outfit}.png'
            if os.path.exists(path):
                outfits[outfit] = cv2.imread(path, -1)
            else:
                # Use default outfit as fallback
                outfits[outfit] = cv2.imread('static/outfits/default.png', -1)
                print(f"Using default outfit for {outfit}")
        return outfits

    def create_default_outfit(self):
        """Create a simple default outfit if none exists"""
        default_path = 'static/outfits/default.png'
        if not os.path.exists(default_path):
            # Create a simple transparent T-shirt shape
            h, w = 400, 300
            img = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Draw a basic T-shirt shape
            pts = np.array([[100,50], [200,50], [220,150], [250,200], 
                          [200,350], [100,350], [50,200], [80,150]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255, 255))
            
            cv2.imwrite(default_path, img)

    def set_current_outfit(self, outfit_data):
        self.current_outfit = outfit_data

    def overlay_clothing(self, frame):
        if self.current_outfit is None:
            return frame

        try:
            outfit_type = self.current_outfit['type']
            if outfit_type not in self.outfits:
                print(f"Warning: Outfit type '{outfit_type}' not available")
                return frame

            frame_h, frame_w = frame.shape[:2]
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                outfit_image = self.outfits[outfit_type]

                # Get body measurements
                keypoints, shoulder_width, body_angle = self.get_body_measurements(landmarks, frame_w, frame_h)

                # Scale and position the T-shirt based on body size
                size_scale = self.size_scales.get(self.current_outfit.get('size', 'M'), 1.0)
                transformed_outfit, dst_points = self.transform_outfit(
                    outfit_image, keypoints, body_angle, size_scale
                )

                # Apply color if specified
                if 'color' in self.current_outfit:
                    transformed_outfit = self.apply_color(transformed_outfit, self.current_outfit['color'])

                # Blend the outfit onto the frame
                frame = self.blend_outfit(frame, transformed_outfit, dst_points, frame_w, frame_h)

        except Exception as e:
            print(f"Error in overlay_clothing: {str(e)}")

        return frame

    def apply_color(self, outfit_image, color_hex):
        try:
            # Convert hex color to RGB
            color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Split channels
            b, g, r, a = cv2.split(outfit_image)
            
            # Create colored mask
            colored = np.zeros_like(outfit_image)
            colored[:, :, 0] = color[2]  # R
            colored[:, :, 1] = color[1]  # G
            colored[:, :, 2] = color[0]  # B
            colored[:, :, 3] = a
            
            # Blend original with color
            gray = cv2.cvtColor(cv2.cvtColor(outfit_image, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
            result = cv2.addWeighted(gray, 0.5, colored, 0.5, 0)
            result[:, :, 3] = a
            
            return result
        except Exception as e:
            print(f"Error applying color: {str(e)}")
            return outfit_image

    def get_body_measurements(self, landmarks, frame_w, frame_h):
        """Get detailed body measurements from landmarks"""
        keypoints = {}
        
        # Upper body points
        keypoints['neck'] = (landmarks[12].x * frame_w + landmarks[11].x * frame_w) / 2, (landmarks[12].y * frame_h + landmarks[11].y * frame_h) / 2
        keypoints['shoulders_left'] = landmarks[11].x * frame_w, landmarks[11].y * frame_h
        keypoints['shoulders_right'] = landmarks[12].x * frame_w, landmarks[12].y * frame_h
        keypoints['chest'] = (landmarks[23].x * frame_w + landmarks[24].x * frame_w) / 2, (landmarks[23].y * frame_h + landmarks[24].y * frame_h) / 2
        
        # Calculate body angles and proportions
        shoulder_width = np.sqrt((keypoints['shoulders_right'][0] - keypoints['shoulders_left'][0])**2 +
                               (keypoints['shoulders_right'][1] - keypoints['shoulders_left'][1])**2)
        
        body_angle = np.arctan2(keypoints['shoulders_right'][1] - keypoints['shoulders_left'][1],
                               keypoints['shoulders_right'][0] - keypoints['shoulders_left'][0])
        
        return keypoints, shoulder_width, body_angle

    def transform_outfit(self, outfit, keypoints, body_angle, size_scale):
        """Apply perspective transform to outfit based on body position"""
        h, w = outfit.shape[:2]
        
        # Calculate transformation points
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Calculate destination points based on body keypoints
        shoulder_width = abs(keypoints['shoulders_right'][0] - keypoints['shoulders_left'][0])
        scaled_width = shoulder_width * size_scale * 1.2  # Add some padding
        
        center_x = (keypoints['shoulders_left'][0] + keypoints['shoulders_right'][0]) / 2
        center_y = keypoints['neck'][1]
        
        # Apply rotation and scaling
        cos_a = np.cos(body_angle)
        sin_a = np.sin(body_angle)
        
        dst_pts = np.array([
            [center_x - scaled_width/2 * cos_a, center_y - scaled_width/2 * sin_a],
            [center_x + scaled_width/2 * cos_a, center_y + scaled_width/2 * sin_a],
            [center_x + scaled_width/2 * cos_a, center_y + h * size_scale],
            [center_x - scaled_width/2 * cos_a, center_y + h * size_scale]
        ], dtype=np.float32)
        
        # Get transformation matrix and apply it
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        transformed = cv2.warpPerspective(outfit, M, (int(scaled_width*1.5), int(h*size_scale*1.2)))
        
        return transformed, dst_pts

    def get_face_measurements(self, frame):
        """Get face measurements using MediaPipe face mesh"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            frame_h, frame_w = frame.shape[:2]
            
            # Get key face points
            nose = (face_landmarks[1].x * frame_w, face_landmarks[1].y * frame_h)
            chin = (face_landmarks[152].x * frame_w, face_landmarks[152].y * frame_h)
            left_ear = (face_landmarks[234].x * frame_w, face_landmarks[234].y * frame_h)
            right_ear = (face_landmarks[454].x * frame_w, face_landmarks[454].y * frame_h)
            
            face_width = np.sqrt((right_ear[0] - left_ear[0])**2 + (right_ear[1] - left_ear[1])**2)
            face_height = np.sqrt((chin[0] - nose[0])**2 + (chin[1] - nose[1])**2)
            face_angle = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
            
            return {
                'center': nose,
                'width': face_width,
                'height': face_height,
                'angle': np.degrees(face_angle)
            }
        return None

    def apply_overlay(self, frame, outfit_image, landmarks, frame_w, frame_h):
        if outfit_image is None or self.current_outfit is None:
            return frame

        try:
            # Get face measurements
            face_data = self.get_face_measurements(frame)
            
            # Get body measurements
            keypoints, shoulder_width, body_angle = self.get_body_measurements(landmarks, frame_w, frame_h)
            
            # Adjust outfit positioning based on both face and body
            if face_data:
                # Use face position to improve outfit placement
                face_scale = face_data['width'] / frame_w
                size_scale = self.size_scales.get(self.current_outfit.get('size', 'M'), 1.0) * face_scale
                outfit_angle = face_data['angle'] + body_angle
                
                # Adjust neck position based on face
                neck_offset = face_data['height'] * 1.2
                keypoints['neck'] = (
                    face_data['center'][0],
                    face_data['center'][1] + neck_offset
                )
            else:
                size_scale = self.size_scales.get(self.current_outfit.get('size', 'M'), 1.0)
                outfit_angle = body_angle
            
            # Transform and apply outfit
            transformed_outfit, dst_points = self.transform_outfit(
                outfit_image, keypoints, outfit_angle, size_scale
            )
            
            # Apply color and blending
            if 'color' in self.current_outfit:
                cache_key = f"{self.current_outfit['type']}_{self.current_outfit['color']}_{outfit_angle}"
                if cache_key not in self._cache:
                    colored_outfit = self.apply_color(transformed_outfit, self.current_outfit['color'])
                    self._cache[cache_key] = colored_outfit
                transformed_outfit = self._cache[cache_key]
            
            # Apply final blending
            frame = self.blend_outfit(frame, transformed_outfit, dst_points, frame_w, frame_h)
            
        except Exception as e:
            print(f"Error in apply_overlay: {str(e)}")
        
        return frame

    def blend_outfit(self, frame, outfit, dst_points, frame_w, frame_h):
        """Enhanced blending with perspective correction"""
        try:
            if outfit.shape[2] == 4:
                y_min, y_max = int(min(dst_points[:, 1])), int(max(dst_points[:, 1]))
                x_min, x_max = int(min(dst_points[:, 0])), int(max(dst_points[:, 0]))
                
                # Ensure coordinates are within frame bounds
                y_min = max(0, y_min)
                y_max = min(frame_h, y_max)
                x_min = max(0, x_min)
                x_max = min(frame_w, x_max)
                
                if y_max > y_min and x_max > x_min:
                    roi = frame[y_min:y_max, x_min:x_max]
                    outfit_roi = cv2.resize(outfit[:, :, :3], (x_max-x_min, y_max-y_min))
                    alpha_roi = cv2.resize(outfit[:, :, 3], (x_max-x_min, y_max-y_min))
                    alpha_roi = np.stack([alpha_roi/255.0] * 3, axis=-1)
                    
                    # Apply perspective-correct blending
                    frame[y_min:y_max, x_min:x_max] = \
                        outfit_roi * alpha_roi + roi * (1 - alpha_roi)
        
        except Exception as e:
            print(f"Error in blend_outfit: {str(e)}")
        
        return frame

    def calculate_inclination(self, point1, point2):
        """Calculate angle of inclination"""
        x1, y1 = point1
        x2, y2 = point2
        return np.degrees(np.arctan2(y2-y1, x2-x1))

    def get_face_boundbox(self, shape, points_range):
        """Get bounding box for face region"""
        coords = shape[points_range]
        return (
            int(np.min(coords[:, 0])),
            int(np.min(coords[:, 1])),
            int(np.max(coords[:, 0]) - np.min(coords[:, 0])),
            int(np.max(coords[:, 1]) - np.min(coords[:, 1]))
        )

    def generate_frames(self):
        with self.camera_lock:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise RuntimeError("Could not access webcam")

        try:
            while True:
                with self.camera_lock:
                    success, frame = self.camera.read()
                if not success:
                    break
                
                processed_frame = self.overlay_clothing(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            self.release_camera()

    def release_camera(self):
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    def process_image(self, file):
        try:
            img_str = file.read()
            nparr = np.frombuffer(img_str, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processed_image = self.overlay_clothing(image)
            _, img_encoded = cv2.imencode('.jpg', processed_image)
            # Convert bytes to base64 string for JSON serialization
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            return jsonify({'image': img_base64})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
