import mediapipe as mp
import cv2
import numpy as np


class TOOL():
    def __init__(self):
        # Initialize the face mesh detector once
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)
        self.MOUTH_INDICES = [
            0, 37, 39, 40, 185, 267, 269, 270, 409,
            78, 80, 81, 82, 191, 291,
            61, 91, 84, 17, 314, 321, 375, 405,
            13, 14, 87, 178, 312, 319, 325,
            88, 95, 146, 181, 308, 310, 311, 317, 402, 415,
        ]


    def set_input_image(self, input_image):
        self.input_image = input_image

    def point_output(self):
        """ Trả về danh sách các điểm miệng đã chuẩn hóa trong bounding box của miệng """
        res = []
        img_rgb = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lấy kích thước ảnh
                width, height = self.input_image.shape[1], self.input_image.shape[0]

                # Đưa tọa độ từ [0,1] về tọa độ thực trên ảnh
                mouth_points = np.array([[face_landmarks.landmark[idx].x * width, face_landmarks.landmark[idx].y * height] for idx in self.MOUTH_INDICES])

                # Tìm bounding box của miệng
                x_min, y_min = np.min(mouth_points, axis=0)
                x_max, y_max = np.max(mouth_points, axis=0)
                box_width, box_height = x_max - x_min, y_max - y_min

                # Tránh chia cho 0 (phòng khi box_width hoặc box_height quá nhỏ)
                if box_width == 0 or box_height == 0:
                    return res  # Trả về danh sách rỗng nếu bounding box không hợp lệ

                # Dịch về gốc của bounding box và chuẩn hóa về [0,1]
                normalized_mouth = (mouth_points - np.array([x_min, y_min])) / np.array([box_width, box_height])

                # Chuyển về dạng list để trả về
                res = normalized_mouth.flatten().tolist()

        return res

    def pic_draw_point(self):
        """ Vẽ các điểm miệng trên ảnh theo tọa độ đã chuẩn hóa với bounding box """
        image_out_point = self.input_image.copy()
        img_rgb = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lấy kích thước ảnh
                width, height = image_out_point.shape[1], image_out_point.shape[0]

                # Đưa tọa độ landmark về hệ ảnh thực
                mouth_points = np.array([[face_landmarks.landmark[idx].x * width,
                                          face_landmarks.landmark[idx].y * height] for idx in self.MOUTH_INDICES])

                # Tìm bounding box của miệng
                x_min, y_min = np.min(mouth_points, axis=0)
                x_max, y_max = np.max(mouth_points, axis=0)
                box_width, box_height = x_max - x_min, y_max - y_min

                if box_width == 0 or box_height == 0:
                    return image_out_point  # Tránh lỗi chia cho 0

                # Dịch về gốc bounding box và chuẩn hóa về [0,1]
                localized_mouth = (mouth_points - np.array([x_min, y_min])) / np.array([box_width, box_height])

                # Vẽ lại các điểm miệng theo tọa độ đã chuẩn hóa
                for point in localized_mouth:
                    cv2.circle(img=image_out_point,
                               center=(
                                   int(point[0] * box_width + x_min), int(point[1] * box_height + y_min)),
                               color=(0, 255, 0), thickness=1, radius=1)

        return image_out_point
    
    def draw_bounding_box(self):
        img_rgb = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        bounding_face = img_rgb
        results = self.face_mesh.process(img_rgb)
        h, w, c = img_rgb.shape
        x_min, x_max, y_min, y_max = [0]*4

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                face_contour_landmarks = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

                x_min = min(face_landmarks.landmark[i].x for i in face_contour_landmarks) * w
                x_max = max(face_landmarks.landmark[i].x for i in face_contour_landmarks) * w
                y_min = min(face_landmarks.landmark[i].y for i in face_contour_landmarks) * h
                y_max = max(face_landmarks.landmark[i].y for i in face_contour_landmarks) * h

                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
                bounding_face = img_rgb[y_min:y_max, x_min:x_max]
                bounding_face = cv2.resize(bounding_face, (w, h))
        return bounding_face
