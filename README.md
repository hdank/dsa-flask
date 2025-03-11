# Data Structure and Algorithms Flask

Ứng dụng trợ lý AI chuyên về Cấu trúc dữ liệu và Thuật toán (Data Structures and Algorithms), được xây dựng trên nền tảng Flask với khả năng truy vấn PDF và hình ảnh.

## Setup

1.  **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Ollama**

    Ensure Ollama server is running

3.  **Run the Flask App**
    ```bash
    python main.py
    ```

## Kiến trúc hệ thống

Dự án được tổ chức theo cấu trúc module như sau:

### Thư mục chính:
- **app/**: Chứa mã nguồn chính của ứng dụng
  - **api/**: Các endpoint API
  - **core/**: Chức năng cốt lõi
  - **service/**: Các dịch vụ trung gian
- **conversations/**: Lưu trữ lịch sử hội thoại
- **db/**: Cơ sở dữ liệu vector và metadata
- **evaluations/**: Kết quả đánh giá câu trả lời
- **pdf/**: Tài liệu PDF được tải lên

## Các chức năng chính

### 1. Tương tác chat sử dụng LLM
Ứng dụng sử dụng các mô hình ngôn ngữ lớn (Large Language Models) để trả lời các câu hỏi về cấu trúc dữ liệu và thuật toán. Hệ thống hỗ trợ đa ngôn ngữ, tự động phát hiện ngôn ngữ tiếng Việt hoặc tiếng Anh trong câu hỏi của người dùng.

### 2. Truy vấn tài liệu PDF
Người dùng có thể tải lên tài liệu PDF để đặt câu hỏi liên quan đến nội dung trong tài liệu. Chức năng này sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) để tìm kiếm và trích xuất thông tin liên quan từ tài liệu.

### 3. Truy vấn hình ảnh
Hệ thống hỗ trợ truy vấn dựa trên hình ảnh, cho phép người dùng tải lên hình ảnh và đặt câu hỏi liên quan đến nội dung trong hình ảnh.

### 4. Đánh giá câu trả lời tự động
Mỗi câu trả lời của hệ thống được đánh giá tự động về cấu trúc và nội dung, đảm bảo chất lượng và định dạng phù hợp.

## API và Endpoints

### Chat API
- **Endpoint**: `/api/chat`
- **Phương thức**: POST
- **Chức năng**: Xử lý các yêu cầu chat AI, tạo hội thoại mới hoặc tiếp tục hội thoại hiện có

### PDF API
- **Endpoint**: `/api/pdf`
- **Phương thức**: POST
- **Chức năng**: Tải lên và xử lý tài liệu PDF

### Query API
- **Endpoint**: `/api/ask_llama`
- **Phương thức**: POST
- **Chức năng**: Truy vấn văn bản đơn giản

### Vision API
- **Endpoint**: `/api/ask_llama_vision`
- **Phương thức**: POST
- **Chức năng**: Truy vấn kết hợp văn bản và hình ảnh

## Các thành phần chính

### Vector Store
Sử dụng cơ sở dữ liệu Chroma để lưu trữ và truy xuất vector nhúng (embeddings) của văn bản. Chức năng này hỗ trợ tìm kiếm ngữ nghĩa trong tài liệu.

### Đánh giá câu trả lời
Hệ thống đánh giá tự động các khía cạnh của câu trả lời:
- **Cấu trúc**: Kiểm tra các thẻ HTML cần thiết (`<CONCEPT>`, `<EXAMPLE>`, `<IMPLEMENTATION>`, v.v.)
- **Nội dung**: Đánh giá chất lượng giải thích, ví dụ, phân tích độ phức tạp thuật toán

### Tạo tiêu đề hội thoại
Hệ thống tự động tạo tiêu đề cho hội thoại dựa trên ngữ cảnh và nội dung, hỗ trợ cả tiếng Việt và tiếng Anh.

## Tính năng đặc biệt

### Định dạng câu trả lời
Câu trả lời được định dạng theo cấu trúc với các phần:
- **Khái niệm** (`<CONCEPT>`)
- **Ví dụ** (`<EXAMPLE>`)
- **Triển khai** (`<IMPLEMENTATION>`)
- **Phân tích độ phức tạp** (`<COMPLEXITY>`)
- **Trực quan hóa** (`<VISUALIZATION>`)
- **Bài tập Lab** (`<LAB>`)
- **Video giáo dục** (`<VIDEOS>`)

### Hỗ trợ đa ngôn ngữ
Hệ thống tự động phát hiện và điều chỉnh nội dung dựa trên ngôn ngữ của người dùng (Tiếng Việt hoặc Tiếng Anh).

## Quy trình xử lý truy vấn

1. Người dùng gửi câu hỏi
2. Hệ thống tự động phát hiện ngôn ngữ
3. Truy xuất tài liệu liên quan (nếu có)
4. Tạo câu trả lời từ LLM
5. Đánh giá và định dạng câu trả lời
6. Gửi kết quả về cho người dùng
7. Lưu trữ hội thoại

## Phát triển và Mở rộng

Dự án được thiết kế theo cấu trúc module, cho phép dễ dàng mở rộng thêm các chức năng mới hoặc thay đổi các thành phần hiện có.