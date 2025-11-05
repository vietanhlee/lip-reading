import cv2
import time
import numpy as np
from detect_mouth import TOOL
from build_model import build
import pickle
from PIL import Image, ImageDraw, ImageFont
import os

categories = pickle.load(open('categories.pkl', 'rb'))

def process(time_step):
    # Load các công cụ cần thiết
    OJ = TOOL()
    
    # Load model tcn (nhớ đăng kí custom layer)
    model = build(189)
    model.load_weights('tcn.weights.h5')  
    # Mở webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Không thể mở camera")
        exit()

    # List chứa các điểm của môi trong 1 khoảng thời gian
    list_mouth_origin= []
    # Nhãn dự đoán gần nhất để hiển thị lên màn hình
    last_label = ""

    # Hàm vẽ chữ tiếng Việt bằng Pillow (hỗ trợ Unicode)
    def draw_vn_text(img_bgr, text, org=(10, 70), font_size=28, color=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0)):
        # Tìm font hỗ trợ tiếng Việt (ưu tiên Arial trên Windows)
        font_paths = [
            r"C:\\Windows\\Fonts\\arial.ttf",
            r"C:\\Windows\\Fonts\\tahoma.ttf",
            r"C:\\Windows\\Fonts\\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font_path = None
        for p in font_paths:
            if os.path.exists(p):
                font_path = p
                break

        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Chuyển sang RGB để dùng Pillow
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Vẽ chữ với viền (stroke) để dễ đọc trên nền phức tạp
        x, y = org
        try:
            draw.text((x, y), text, font=font, fill=tuple(int(c) for c in color),
                      stroke_width=stroke_width, stroke_fill=tuple(int(c) for c in stroke_fill))
        except TypeError:
            # Fallback cho Pillow cũ không hỗ trợ stroke
            draw.text((x, y), text, font=font, fill=tuple(int(c) for c in color))

        # Chuyển ngược về BGR cho OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


    while True:
        start_time = time.time()  # Lấy thời gian bắt đầu frame

        # Đọc ảnh từ webcam
        check, frame = cam.read()
        if not check:
            break
        # Đảo ngược ảnh
        frame = cv2.flip(frame, 1)

        # set ảnh đầu vào cho TOOL
        OJ.set_input_image(frame)
        
        # Cắt bớt phần đầu list khi độ dài quá time_step
        while len(list_mouth_origin) >= time_step:
            list_mouth_origin = list_mouth_origin[1:]
        # Lấy các điểm của môi
        mouth = OJ.point_output()
        # Thêm list điểm môi mới vào cuối list chứa các điểm môi
        list_mouth_origin.append(mouth)

        res = [['none']] # Biến flag 
        
        # Chỉ khi số lượng frame đủ yêu cầu thì mới predict
        if(len(list_mouth_origin) == time_step):
            # Bắt lỗi khi dự đoán do dữ liệu đầu vào không hợp lệ (không đủ số lượng hoặc không đúng kích thước)
            try:
                # Biến đổi dữ liệu đầu vào 
                arr_mouth = np.array(list_mouth_origin)
                # arr_mouth = arr_mouth.reshape(-1, 40, 2)
                arr_mouth = np.expand_dims(arr_mouth, axis= 0)
                
                # Tiến hành dự đoán
                res = model.predict(arr_mouth, verbose = False)
                # print(res)
                res = categories[np.argmax(res[0], axis= 0)]
                print("Dự đoán: ", res)
                # Cập nhật nhãn để hiển thị lên màn hình
                last_label = str(res)
                if last_label[0] == '1':
                    last_label = 'Tục'
                
            except Exception as e:
                print('Lỗi khi dự đoán', e)
                # pass

        # Ảnh đầu ra với các điểm đã vẽ
        frame_out = OJ.pic_draw_point()
        # Tính FPS
        fps = 1 / (time.time() - start_time)
        # Hiển thị FPS (ASCII - không dấu vẫn dùng OpenCV)
        cv2.putText(frame_out, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Hiển thị nhãn dự đoán (tiếng Việt có dấu) bằng Pillow
        if last_label:
            frame_out = draw_vn_text(frame_out, f"Dự đoán: {last_label}", (10, 70),
                                      font_size=28, color=(0, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        
       
        # Hiển thị màn hình
        cv2.imshow("out_put", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Chạy chương trình
    # NHớ thay đổi tham số time_step để thay đổi số lượng điểm môi cần dự đoán
    process(32)