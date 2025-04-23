# Sau đây là cấu trúc project
  ```
    DATA
        |
        |__Mã sinh viên
                |
                |___DATA_TUC
                |      |
                |      |__.csv
                |      |
                |      |__.wav
                |
                |___DATA_KHONG_TUC
                        |
                        |__.csv
                        |
                        |__.wav
```

- Thư mục Data chính là tổ hợp các file `csv` chứa data về các điểm miệng được lưu khi tạo data, mỗi dòng là 40 giá trị, cứ 2 giá trị là thông tin 1 point miệng

- File `collecting.py` là file lấy data (dùng để chạy lấy DATA)

- File `main.py` là file để chạy

- File `tool` chỉ là một file công cụ chứa 2 hàm là lấy ra dữ liệu point (gồm 40 giá trị và cứ 2 giá trị chính là thông tin của một điểm point miệng)

- File `training` dùng để training data thôi, đọc kĩ các dòng code, có thể thay đổi các tham số theo ý thích (time_step nên để từ 20 đến 40 thôi cao hơn hay ít hơn thì không nên do không quá nhạy, để mặc định là 32). 

# Traing data
- Đầu tiên cần có DATA (tức thư mục DATA cần phải có, chưa có thì chạy file `collecting.py` để lấy)

- Máy có GPU thì chạy luôn file `training.ipynb`. Tốt nhất là chạy phần preprocessing trong file rồi lấy 2 file `.pkl` rồi import colab để training. Nếu dùng [Google Colab](https://colab.research.google.com/drive/10MGuuBpTkuUrABmeYWCbGe5di9wKN2jj?usp=sharing) để training thì tải model và chạy file `main.py` dùng