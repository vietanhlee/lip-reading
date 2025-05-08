# Cơ sở lý thuyết


# Sau đây là cấu trúc project
- Thư mục Data chính là tổ hợp các file `csv` chứa data về các điểm miệng được lưu khi tạo data, mỗi dòng là 40 giá trị, cứ 2 giá trị là thông tin 1 point miệng

- Cấu trúc thư mục DATA
  ```
    DATA
        |
        |__ Folder tên người
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


- File `collecting.py` là file lấy data (dùng để chạy lấy DATA)

- File `main.py` là file để chạy (khi có model)

- File `tool.py` chỉ là một file công cụ chứa 2 hàm là lấy ra dữ liệu point (gồm 40 giá trị và cứ 2 giá trị chính là thông tin của một điểm point miệng)

- File `TCN.ipynb` dùng để training data dựa trên model `TCN` 
- File `LSTM.ipynb` dùng để training data dựa trên model `LSTM`
> Có thể thay đổi các tham số theo ý thích (time_step nên để từ 20 đến 40 thôi cao hơn hay ít hơn thì không nên do không quá nhạy, để mặc định là 32). 

# Traing data
- Đầu tiên cần có DATA (tức thư mục DATA cần phải có, chưa có thì chạy file `collecting.py` để lấy)

- Nếu thiết bị đủ mạnh thì có thể training file `TCN.ipynb` trên local. luôn. Tốt nhất là chạy phần preprocessing trong file đó để xuất 2 tệp `X.pkl` và `Y.pkl` rồi import vào [Google Colab](https://colab.research.google.com/drive/10MGuuBpTkuUrABmeYWCbGe5di9wKN2jj?usp=sharing) để training và tải tệp `tcn.keras` về máy và chạy thử file `main.py`

> Tương tự với file `LSTM`

# So sánh với LSTM
