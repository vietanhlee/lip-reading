import cv2
import csv
import os
from detect_mouth import TOOL
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Gọi tool lấy data points
tool = TOOL()

os.makedirs('DATA_KHONG_TUC', exist_ok= True)
os.makedirs('DATA_TUC', exist_ok= True)

# Hàm trộn từ (có lưu vào file csv mới)
def process_combine_words():
    # Đầu ra là mảng 2D
    bad_word = pd.read_csv(r"bad_words.csv")
    good_word = pd.read_csv(r"good_words.csv")

    padding_bad = np.full(shape=(bad_word.shape), fill_value= '1')
    padding_good = np.full(shape=(good_word.shape), fill_value= '0')

    bad = bad_word.to_numpy() + padding_bad
    good = good_word.to_numpy() + padding_good

    data = np.concatenate([bad, good])

    # Trộn hoán vị và duỗi thẳng array
    words_combine = np.random.permutation(data).flatten()

    return words_combine

# Đọc danh sách từ
data_words = process_combine_words()
current_index = 0

# Các biến lính canh
Is_collecting = True
saving = False

# Đường dẫn đến font hỗ trợ tiếng Việt
font_path = "arial.ttf"
font_size = 30
font = ImageFont.truetype(font_path, font_size)

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('Mouth Points')

while Is_collecting and current_index < len(data_words):
    # Thư mục lưu dữ liệu mặc định
    DATA_DIR = r"DATA_TUC"

    word_origin = data_words[current_index]
    if(word_origin[-1] == '0'): DATA_DIR = r"DATA_KHONG_TUC"

    word = word_origin[:-1].strip()

    file_name = os.path.join(DATA_DIR, f'{word}.csv')
    video_file = os.path.join(DATA_DIR, f'{word}.avi')

    # Lấy thông số video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20  # Số khung hình trên giây

    # Định dạng codec và tạo đối tượng VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        print(f"Đang chuẩn bị thu thập dữ liệu cho từ: {word}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_cp = frame.copy()

            frame = cv2.flip(frame, 1)
            tool.set_input_image(frame)  # Truyền ảnh vào công cụ TOOL
            mouth_points = tool.point_output()  # Lấy ra các điểm ảnh của miệng

            # Lấy hình ảnh với các điểm miệng đã được vẽ
            image_with_points = tool.pic_draw_point()

            # Chuyển image_with_points từ OpenCV (BGR) sang PIL (RGB)
            image_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)

           # Vẽ văn bản lên ảnh chính (frame)
            draw.text((50, 10), "Nhấn s mỗi khi nói ",
                      font=font, fill=(120, 120, 0))  # Đỏ
            draw.text((50, 40), f"Nói: {word}",
                      font=font, fill=(255, 0, 0))  # Xanh lá
            draw.text((50, 70), "Nhấn d sang từ tiếp theo",
                      font=font, fill=(0, 0, 255))  # Xanh dương
            draw.text((50, 100), f"Nhấn b để nói lại từ trước đó",
                      font=font, fill=(0, 0, 225))  # Xanh lá
            
            # Chuyển lại từ PIL sang OpenCV
            image_with_points = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            bounding_face = tool.draw_bounding_box()
            bounding_face = cv2.cvtColor(bounding_face, cv2.COLOR_BGR2RGB)

            # Hiển thị image_with_points với text trên cửa sổ Mouth Points
            cv2.imshow("Mouth Points", image_with_points)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                saving = True
                print(f"Đang thu thập cho từ: {word}")

            elif key == ord('d'):
                saving = False
                current_index += 1
                print(f"Hoàn tất lưu dữ liệu cho từ: {word}. Chuyển sang từ tiếp theo...\n")

                break
            elif key == ord("b"):
                if current_index > 0:
                    current_index -= 1
                print("Đã lại từ trước đó")
                break
            elif key == ord('q'):  # Nhấn 'q' để thoát toàn bộ chương trình
                print("Dừng thu thập dữ liệu.")
                Is_collecting = False
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                exit()

            if saving:
                if mouth_points:
                    writer.writerow(mouth_points)  # Lưu dữ liệu vào CSV
                    out.write(bounding_face)  # Ghi frame vào video (frame gốc từ webcam)
        

    out.release()  # Đóng file video
    
cap.release()
cv2.destroyAllWindows()