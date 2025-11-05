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

> Data qua xử lý [tải ở đây](https://drive.google.com/drive/folders/1lZtWnx8I1sMJzvITtP0goEwO-RInpCoG?usp=drive_link) 

# Traing data
- Đầu tiên cần có DATA (tức thư mục DATA cần phải có, chưa có thì chạy file `collecting.py` để lấy)

- Nếu thiết bị đủ mạnh thì có thể training file `TCN.ipynb` trên local luôn. Tốt nhất chạy trên [Google Colab](https://colab.research.google.com/drive/10MGuuBpTkuUrABmeYWCbGe5di9wKN2jj?usp=sharing) để training và tải tệp `tcn.keras` về máy và chạy thử file `main.py`

> Tải luôn model [tại đây](https://drive.google.com/file/d/1rsibV_h-EPq5GQKT4KA4CZ7-OFtghoNq/view?usp=sharing)

# So sánh kết quả training giữa LSTM với TCN
## Khi sử dụng LSTM
![](https://raw.githubusercontent.com/vietanhlee/lip-reading/refs/heads/main/LSTM.png)

- Từ đồ thị ta thấy model overfiting mặc dù training với 100 epochs

> Tải model [tại đây](https://drive.google.com/file/d/1hWFf94gVuSo-RHy0kvbP6hKhTGqYi0ok/view?usp=sharing)

## Khi sử dụng TCN
![](https://raw.githubusercontent.com/vietanhlee/lip-reading/refs/heads/main/TCN.png)

- Biểu đồ này đã hợp cho việc predict realtime

> Tải model [tại đây](https://drive.google.com/file/d/1rsibV_h-EPq5GQKT4KA4CZ7-OFtghoNq/view?usp=sharing)

# Tham khảo

[Paper TCN](https://arxiv.org/pdf/1803.01271)

[Code TCN theo pytorch](https://github.com/locuslab/TCN)