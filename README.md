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

#### Chức năng chính của ChromaDB
1. **Lưu trữ tài liệu**: Chuyển đổi văn bản thành vector nhúng và lưu trữ
2. **Phân đoạn tài liệu**: Chia tài liệu thành các đoạn nhỏ (chunks) để xử lý hiệu quả
3. **Tìm kiếm tương tự**: Truy xuất các đoạn tài liệu có ngữ nghĩa tương đồng với truy vấn
4. **Quản lý metadata**: Lưu trữ thông tin về nguồn gốc và thuộc tính của tài liệu

#### Công thức toán học
Vector embedding sử dụng mô hình FastEmbed để chuyển đổi văn bản thành vector có kích thước cố định. Khi tìm kiếm, ChromaDB sử dụng Approximate Nearest Neighbors (ANN) với công thức:

$ANN(q, D) = \arg\max_{d \in D} similarity(q, d)$

Trong đó:
- $q$ là vector truy vấn
- $D$ là tập hợp các vector tài liệu
- $similarity$ là hàm đo độ tương đồng (thường là cosine similarity)

#### Triển khai trong dự án
```python
def store_documents(docs):
    # Tạo ID tài liệu duy nhất
    document_id = str(uuid.uuid4())
    
    # Phân đoạn tài liệu thành chunks
    chunks = text_splitter.split_documents(docs)
    
    # Tạo ID duy nhất cho mỗi chunk
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    
    # Lưu vào ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_FOLDER,
        ids=chunk_ids,
    )
    vector_store.persist()
    
    return document_id
```

#### Cải tiến với Dynamic-K
Hệ thống sử dụng cơ chế Dynamic-K để điều chỉnh số lượng kết quả truy vấn dựa trên độ phức tạp của truy vấn, giúp cân bằng giữa chất lượng thông tin và hiệu suất xử lý.
```python
def get_dynamic_k(query: str, base_k: int = 3, words_per_increment: int = 10, max_increment: int = 3) -> int:
    word_count = len(query.split())
    increments = min(word_count // words_per_increment, max_increment)
    return base_k + increments
```
### Retrieval Augmented Generation (RAG)

RAG (Retrieval Augmented Generation) được triển khai để nâng cao chất lượng câu trả lời bằng cách kết hợp khả năng truy xuất thông tin từ tài liệu với khả năng sinh nội dung của mô hình ngôn ngữ lớn.

#### Cơ chế hoạt động
1. **Nhúng truy vấn**: Khi người dùng gửi truy vấn, hệ thống chuyển đổi câu hỏi thành vector nhúng (embedding)
2. **Truy xuất ngữ cảnh**: Hệ thống tìm kiếm các tài liệu liên quan trong vector store
3. **Bổ sung ngữ cảnh**: Thông tin được trích xuất được kết hợp với prompt gửi đến LLM
4. **Sinh câu trả lời**: LLM tạo câu trả lời dựa trên cả truy vấn và ngữ cảnh đã truy xuất

#### Công thức toán học
RAG sử dụng độ tương đồng cosine để so sánh vector truy vấn với vector tài liệu:

$similarity(q, d) = \frac{q \cdot d}{||q|| \times ||d||}$

Trong đó:
- $q$ là vector nhúng của truy vấn
- $d$ là vector nhúng của tài liệu
- $q \cdot d$ là tích vô hướng của hai vector
- $||q||$ và $||d||$ là độ dài Euclidean của vector

### Triển khai trong dự án
```python
def retrieve_relevant_documents(query, vector_store, base_k: int = 3) -> list:
    @lru_cache(maxsize=128)
    def _cached_retrieval(query_hash, base_k):
        dynamic_k = get_dynamic_k(query, base_k=base_k)
        docs_and_scores = vector_store.similarity_search_with_score(query, k=dynamic_k)
        # Chuyển đổi kết quả thành định dạng phù hợp
        # ...
        return docs
    
    # Sử dụng caching để cải thiện hiệu suất
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return _cached_retrieval(query_hash, base_k)
```

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

### Quy trình RAG chi tiết

1. **Tiền xử lý tài liệu**:
   - Phân tích tài liệu PDF
   - Chia thành các đoạn nhỏ (chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
   - Tạo vector nhúng cho mỗi đoạn với FastEmbedEmbeddings
   - Lưu vào ChromaDB với metadata

2. **Xử lý truy vấn**:
   - Tạo vector nhúng từ truy vấn người dùng
   - Tìm kiếm các đoạn tài liệu tương tự trong ChromaDB
   - Kết hợp thông tin từ lịch sử hội thoại (nếu có)
   - Xây dựng prompt với ngữ cảnh trích xuất
   - Gửi đến LLM để tạo câu trả lời

3. **Tối ưu hóa**:
   - Cache kết quả truy vấn cho các truy vấn tương tự
   - Điều chỉnh số lượng kết quả trả về dựa trên độ phức tạp của truy vấn
   - Đánh giá và định dạng câu trả lời theo cấu trúc chuẩn

Sự kết hợp của RAG và ChromaDB tạo nên một hệ thống hỏi đáp thông minh, có khả năng truy xuất thông tin từ tài liệu DSA và đưa ra câu trả lời có cấu trúc, đầy đủ thông tin một cách tự động.

## Phát triển và Mở rộng

Dự án được thiết kế theo cấu trúc module, cho phép dễ dàng mở rộng thêm các chức năng mới hoặc thay đổi các thành phần hiện có.