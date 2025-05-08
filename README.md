# Cơ sở lý thuyết
## Tổng quan về bài toán Time Series

Time series (chuỗi thời gian) là dạng dữ liệu được thu thập theo trình tự thời gian, thường thấy trong tài chính, y tế, công nghiệp, v.v. Đặc điểm chính là tính phụ thuộc thời gian và tính không dừng. Nhiệm vụ phổ biến gồm dự báo (forecasting), phát hiện bất thường (anomaly detection), phân loại.

---

## Một số phương pháp giải quyết Time Series

### 1. ARIMA
- **Ưu**: Dễ triển khai, hiệu quả với chuỗi tuyến tính.
- **Nhược**: Không tốt với dữ liệu phi tuyến hoặc nhiều biến.

### 2. LSTM
- **Ưu**: Mô hình hóa được quan hệ dài hạn, học phi tuyến tốt.
- **Nhược**: Chậm, dễ gặp vấn đề gradient, khó tối ưu.

### 3. Transformer (Informer, Autoformer)
- **Ưu**: Tốt với chuỗi dài, có tính mùa vụ.
- **Nhược**: Tốn tài nguyên, cần xử lý vị trí cẩn thận.

### 4. TCN (Temporal Convolutional Network) — *Nổi bật*

TCN dùng mạng tích chập để mô hình hóa chuỗi thời gian với các đặc trưng chính:
- **Causal Convolution**: Đảm bảo đầu ra chỉ phụ thuộc đầu vào quá khứ.
- **Dilated Convolution**: Mở rộng phạm vi nhìn mà không tăng số tầng.
- **Residual Connection**: Giúp huấn luyện ổn định hơn.

**Ưu điểm:**
- Huấn luyện song song nhanh hơn RNN.
- Mô hình hóa chuỗi dài hiệu quả.
- Ít gặp lỗi gradient vanish.

**Nhược điểm:**
- Cần nhiều lớp/dilation lớn cho chuỗi rất dài.
- Kém hiệu quả nếu phụ thuộc không tuần tự rõ ràng.

**Ứng dụng:** Dự báo nhu cầu, phát hiện bất thường, cảm biến IoT, v.v.

## Kết luận

TCN là lựa chọn hiệu quả, hiện đại cho bài toán chuỗi thời gian, đặc biệt với chuỗi dài và cần huấn luyện nhanh. Có tiềm năng thay thế LSTM trong nhiều ứng dụng thực tế.

---
# Cấu trúc project
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
