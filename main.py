# Import các thư viện cần thiết
import cv2
import time
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from detect_mouth import TOOL
from tensorflow.keras.utils import custom_object_scope

# Import lớp tùy chỉnh của bạn
from modul import TemporalBlock, TemporalConvNet  # Đảm bảo thay thế 'your_module' bằng module chứa lớp TemporalBlock

def process(time_step):
    # Load các công cụ cần thiết
    OJ = TOOL()
    
    model = load_model("tcn.keras", custom_objects={"TemporalBlock": TemporalBlock, "TemporalConvNet": TemporalConvNet})

    # Mở webcam
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Không thể mở camera")
        exit()

    # List chứa các điểm của môi trong 1 khoảng thời gian
    list_mouth_origin= []

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

        # Dự đoán
        res = [['none']]
        if(len(list_mouth_origin) == time_step):
            # bắt lỗi khi dự đoán do dữ liệu đầu vào không hợp lệ (không đủ số lượng hoặc không đúng kích thước)
            try:
                arr_mouth = np.array(list_mouth_origin)
                arr_mouth = arr_mouth.reshape(-1, 40, 2)
                arr_mouth = np.expand_dims(arr_mouth, axis= 0)
                res = model.predict(arr_mouth, verbose = False)
                # print(res)
            except:
                print('Lỗi khi dự đoán', arr_mouth)
                pass

        # Ảnh đầu ra với các điểm đã vẽ
        frame_out = OJ.pic_draw_point()
        # Tính FPS
        fps = 1 / (time.time() - start_time)
        # Hiển thị FPS
        cv2.putText(frame_out, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Hiển thị kết quả
        color = None
        if(res != [['none']]):
            res = round(res[0][0], 2)
            if res < 0.5:
                res = str(res) + " " + "khong tuc"
                color = (0, 255, 0)
            else:
                res = str(res) + " " + "tuc"
                color = (0, 0, 255)
        cv2.putText(frame_out, f"res: {res}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

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