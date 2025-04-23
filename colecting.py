import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import csv
import cv2
from PIL import Image, ImageDraw, ImageFont
from detect_mouth import TOOL

tool = TOOL()

def run_main_process(parent_dir):
    bad_word = pd.read_csv("bad_words.csv")
    good_word = pd.read_csv("good_words.csv")

    padding_bad = np.full(shape=(bad_word.shape), fill_value='1')
    padding_good = np.full(shape=(good_word.shape), fill_value='0')

    bad = bad_word.to_numpy() + padding_bad
    good = good_word.to_numpy() + padding_good

    data = np.concatenate([bad, good])
    words_combine = np.random.permutation(data).flatten()

    current_index = 0
    Is_collecting = True
    saving = False

    font_path = "arial.ttf"
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Mouth Points')

    while Is_collecting and current_index < len(words_combine):
        check_write = False
        word_origin = words_combine[current_index]
        if word_origin[-1] == '0':
            DATA_DIR = os.path.join(parent_dir, "DATA_KHONG_TUC")
        else:
            DATA_DIR = os.path.join(parent_dir, "DATA_TUC")

        word = word_origin[:-1].strip()
        file_name = os.path.join(DATA_DIR, f'{word}.csv')
        video_file = os.path.join(DATA_DIR, f'{word}.avi')

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_file, fourcc, fps,
                              (frame_width, frame_height))

        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            print(f"Đang chuẩn bị thu thập dữ liệu cho từ: {word}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                tool.set_input_image(frame)
                mouth_points = tool.point_output()
                image_with_points = tool.pic_draw_point()

                image_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(pil_image)

                draw.text((50, 10), "Nhấn s mỗi khi nói ",
                          font=font, fill=(120, 120, 0))
                draw.text((50, 40), f"Nói: {word}",
                          font=font, fill=(255, 0, 0))
                draw.text((50, 70), "Nhấn d sang từ tiếp theo",
                          font=font, fill=(0, 0, 255))
                draw.text((50, 100), "Nhấn b để nói lại từ trước đó",
                          font=font, fill=(0, 0, 225))

                image_with_points = cv2.cvtColor(
                    np.array(pil_image), cv2.COLOR_RGB2BGR)
                bounding_face = tool.draw_bounding_box()
                bounding_face = cv2.cvtColor(bounding_face, cv2.COLOR_BGR2RGB)

                cv2.imshow("Mouth Points", image_with_points)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    saving = True
                    print(f"Đang thu thập cho từ: {word}")
                elif key == ord('d'):
                    saving = False
                    current_index += 1
                    print(
                        f"Hoàn tất lưu dữ liệu cho từ: {word}. Chuyển sang từ tiếp theo...\n")
                    break
                elif key == ord("b"):
                    if current_index > 0:
                        current_index -= 1
                    print("Đã quay lại từ trước đó")
                    break
                elif key == ord('q'):
                    print("Dừng thu thập dữ liệu.")
                    Is_collecting = False
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    return  # Thoát luôn

                if saving and mouth_points:
                    writer.writerow(mouth_points)
                    out.write(bounding_face)
                    check_write = True

            if not check_write:
                current_index -= 1
        out.release()

    cap.release()
    cv2.destroyAllWindows()

def create_folder():
    folder_name = entry.get().strip()
    if folder_name:
        parent_dir = os.path.join(
            'DATA', folder_name)
        try:
            os.makedirs(os.path.join(
                parent_dir, 'DATA_KHONG_TUC'), exist_ok=True)
            os.makedirs(os.path.join(parent_dir, 'DATA_TUC'), exist_ok=True)
            # messagebox.showinfo("Thành công", f"Đã tạo thư mục:\n{parent_dir}")
            entry.delete(0, tk.END)
            root.withdraw()  # Ẩn cửa sổ GUI trong lúc thu thập
            run_main_process(parent_dir)
            root.deiconify()  # Hiện lại sau khi thu thập xong
        except Exception as e:
            pass
    else:
        messagebox.showwarning("Thông báo", "Vui lòng nhập tên thư mục.")


def stop_program():
    root.destroy()


# Giao diện GUI
root = tk.Tk()
root.title("Tạo thư mục và thu thập dữ liệu")
root.geometry("500x200")
tk.Label(root, text="Nhập mã sinh viên của bạn").pack(pady=5)

entry = tk.Entry(root, width=30)
entry.pack(pady=5)
entry.focus()
entry.config(font=("Arial", 14), fg="blue", bg="lightyellow", justify="center")

tk.Button(root, text="Tạo thư mục & bắt đầu thu thập",
          command=create_folder).pack(pady=5)
tk.Button(root, text="Dừng chương trình", command=stop_program,
          fg="white", bg="red").pack(pady=5)

root.mainloop()
