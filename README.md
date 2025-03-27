# Sau đây là cấu trúc project
- Hai thư mục có tên đầu là Data chính là tổ hợp các file `csv` chứa data về các điểm miệng được lưu khi ae tạo data, mỗi dòng là 40 giá trị, cứ 2 giá trị là thông tin 1 point miệng

- File `Collecting` là file lấy data, nhớ thay `start_gui(0)` thành `start_gui(1)` để lấy data tục còn không thì không cần

- File `main` là file để chạy thôi, có comment đầy đủ rồi, đọc dễ hiểu

- File `tool` chỉ là một file công cụ chứa 2 hàm là lấy ra dữ liệu point (gồm 40 giá trị và cứ 2 giá trị chính là thông tin của một điểm point miệng)

- File `training` dùng để training data thôi, đọc kĩ các dòng code, có thể thay đổi các tham số theo ý thích (time_step nên để từ 20 đến 40 thôi cao hơn hay ít hơn thì không nên do không quá nhạy)

