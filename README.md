# Data Structure and Algorithms Flask

Ứng dụng trợ lý AI chuyên về Cấu trúc dữ liệu và Thuật toán (Data Structures and Algorithms), được xây dựng trên nền tảng Flask với khả năng truy vấn PDF và hình ảnh.

## Table of Contents
- [Setup](#setup)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Các chức năng chính](#các-chức-năng-chính)
- [API và Endpoints](#api-và-endpoints)
- [Các thành phần chính](#các-thành-phần-chính)
- [Tính năng đặc biệt](#tính-năng-đặc-biệt)
- [Quy trình xử lý truy vấn](#quy-trình-xử-lý-truy-vấn)
- [Function Calling](#function-calling)
- [Hệ thống Đánh giá Tự động](#hệ-thống-đánh-giá-tự-động)
- [Phát triển và Mở rộng](#phát-triển-và-mở-rộng)

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

#### Triển khai trong dự án
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

## Function Calling
Hệ thống DSA Flask triển khai cơ chế Function Calling cho phép mô hình ngôn ngữ lớn tương tác với các công cụ bên ngoài, truy xuất thông tin, và thực hiện các tác vụ phức tạp.

### Tổng quan
Function Calling là một cơ chế giúp LLM xác định khi nào và cách thức gọi các hàm được định nghĩa trước. Cơ chế này cho phép mô hình mở rộng khả năng của mình bằng cách:

- Tìm kiếm thông tin từ nguồn bên ngoài
- Thực hiện các tính toán phức tạp
- Tạo biểu diễn trực quan
- Tương tác với các API và dịch vụ khác

### Cấu trúc và Quy trình
1. **Đăng ký hàm**: Các hàm được đăng ký trong module `function_call.py` để LLM có thể truy cập
2. **Chuẩn bị định nghĩa**: Hệ thống tạo các định nghĩa hàm ở định dạng tương thích với API của LLM
3. **Gửi yêu cầu**: Định nghĩa hàm và tin nhắn người dùng được gửi đến API
4. **Phân tích phản hồi**: Nếu LLM quyết định gọi hàm, hệ thống xác định hàm cần gọi và tham số
5. **Thực thi hàm**: Hệ thống gọi hàm tương ứng với các tham số được cung cấp
6. **Phản hồi kết quả**: Kết quả của hàm được đưa vào tin nhắn và gửi lại cho API
7. **Lặp lại hoặc kết thúc**: Quá trình được lặp lại nếu cần thiết hoặc kết thúc với phản hồi cuối cùng

### Các hàm công cụ chính

#### 1. `get_education_video`
Tìm kiếm video giáo dục về các chủ đề DSA, hỗ trợ cả tiếng Việt và tiếng Anh (ưu tiên các video được hardcode):
```python
def get_educational_video(topic: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Tìm kiếm video giáo dục về một chủ đề DSA cụ thể.
    
    Args:
        topic: Chủ đề cần tìm video (ví dụ: "bubble sort", "sắp xếp nổi bọt")
        max_results: Số lượng video tối đa cần trả về
        
    Returns:
        List[Dict[str, str]]: Danh sách video với thông tin title, url, thumbnail_url, channel_name
    """
```

#### 2. `_fetch_videos_from_api`
Tìm kiếm video từ YouTube API nếu video hardcode không đáp ứng được yêu cầu câu hỏi người dùng
```python
def _fetch_videos_from_api(topic: str, max_results: int) -> List[Dict[str, str]]:
    """
    Fall back function to get videos from YouTube API if no recommended videos are found.
    
    Args:
        topic (str): The topic to search for
        max_results (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, str]]: List of video information dictionaries
    """
```

### Chuyển đổi thuật ngữ DSA Tiếng Việt-Anh
Hệ thống hỗ trợ chuyển đổi giữa thuật ngữ DSA tiếng Việt và tiếng Anh, giúp chức năng tìm kiếm video hoạt động hiệu quả với người dùng của cả hai ngôn ngữ. Bộ chuyển đổi thuật ngữ bao gồm:
```python
    language_mapping = {
        "sắp xếp chọn": "selection sort",
        "sắp xếp nhanh": "quick sort",
        "sắp xếp nổi bọt": "bubble sort",
        "sắp xếp chèn": "insertion sort",
        "sắp xếp trộn": "merge sort",
        "tìm kiếm tuyến tính": "linear search",
        "tìm kiếm nhị phân": "binary search",
        "danh sách liên kết đơn": "single linked list",
        "danh sách liên kết kép": "double linked list",
        "danh sách liên kết vòng": "circular linked list",
        "ngăn xếp": "stack",
        "hàng đợi": "queue",
        "hàng đợi ưu tiên": "priority queue",
        "bảng băm": "hash table",
        "tìm kiếm theo chiều sâu": "depth first search",
        "tìm kiếm theo chiều rộng": "breadth first search",
        "cây nhị phân": "binary tree"
    }
```
Điều này đảm bảo rằng người dùng có thể yêu cầu thông tin về các thuật toán và cấu trúc dữ liệu bằng thuật ngữ tiếng Việt, và hệ thống vẫn có thể tìm kiếm video giáo dục phù hợp.

## Hệ thống Đánh giá Tự động
DSA Flask tích hợp hệ thống đánh giá tự động để phân tích và đánh giá chất lượng câu trả lời, đảm bảo thông tin chính xác và hữu ích cho người dùng.

### Các tiêu chí đánh giá
Hệ thống đánh giá dựa trên 3 khía cạnh chính:

1. **Cấu trúc** - Đánh giá tính đầy đủ và cấu trúc phù hợp của câu trả lời
2. **Nội dung** - Đánh giá chất lượng nội dung thực tế
3. **Độ liên quan** - Đánh giá mức độ phù hợp với câu hỏi

### Thuật toán và công thức đánh giá
#### 1. Đánh giá cấu trúc

Hệ thống đánh giá sự hiện diện và nội dung của các thẻ HTML bắt buộc trong câu trả lời:

- `<CONCEPT>`: Giải thích khái niệm
- `<EXAMPLE>`: Ví dụ minh họa
- `<IMPLEMENTATION>`: Triển khai mã nguồn
- `<COMPLEXITY>`: Phân tích độ phức tạp
- `<LAB>`: Phần thực hành (nếu có)

Điểm số được tính như sau:
- Thẻ có nội dung đầy đủ: +10-15 điểm
- Thẻ tồn tại nhưng trống: +5 điểm
- Thẻ bị thiếu hoặc lỗi: 0 điểm
- Cấu trúc tổng thể: tối đa 100 điểm

#### 2. Đánh giá nội dung

Đánh giá chất lượng của nội dung trong câu trả lời:

**Chất lượng giải thích khái niệm** (30% trọng số): 
```python
concept_quality = quality_of_explanation(concept_content)
```

**Chất lượng ví dụ** (20% trọng số): 
```python
example_quality = quality_of_example(example_content)
```

**Chất lượng mã nguồn** (20% trọng số): 
```python
comment_ratio = comment_lines / max(1, code_lines)
```

**Phân tích độ phức tạp** (30% trọng số):
- Phân tích đầy đủ độ phức tạp thuật toán: +25 điểm
- Phân tích một phần: +15 điểm
- Phân tích cơ bản: +5 điểm

**Điều chỉnh theo độ dài**:
- ≥500 từ: điểm * 1.1
- <150 từ: điểm * 0.8

#### 3. Đánh giá độ liên quan
Đánh giá mức độ phù hợp của câu trả lời với câu hỏi:

**Tỷ lệ từ khóa chung** (40% trọng số):
```python
keyword_ratio = len(common_keywords) / max(1, len(query_keywords)) keyword_score = min(100, keyword_ratio * 100)
```

**Tương đồng ngữ nghĩa** (40% trọng số):
```python
semantic_score = cosine_similarity(query_embedding, response_embedding) * 100
```

**Nhận dạng chủ đề DSA** (10% trọng số):
```python
topic_score = 100 if detected_dsa_topics else 0
```

**Kiểm tra định dạng phản hồi** (10% trọng số):
```pythonformat_score = 100 if format_matches else 0
```

### Đo lường tương đồng văn bản
Hệ thống sử dụng nhiều phương pháp khác nhau để đo lường độ tương đồng văn bản:

1. **Tỷ lệ tương đồng chuỗi (SequenceMatcher)**:
Phương pháp này so sánh trực tiếp các ký tự giữa hai chuỗi và trả về tỷ lệ khớp từ 0 đến 100%.

```python
string_similarity = SequenceMatcher(None, expected_output, completion).ratio() * 100
```

2. **Tỷ lệ tương đồng từ (Jaccard Similarity)** :
Tính tỷ lệ số từ chung giữa hai văn bản so với tổng số từ của văn bản dài hơn.

```python
word_similarity = len(common_words) / max(len(expected_words), len(completion_words)) * 100
```
Tương đồng Jaccard đo lường mức độ chồng lấp giữa hai tập hợp bằng cách tính tỷ lệ giữa giao điểm và hợp của hai tập. Công thức toán học:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Trong đó:

* $A$ và $B$ là hai tập hợp từ (được trích xuất từ hai văn bản)
* $|A \cap B|$ là số phần tử trong giao của hai tập (số từ chung)
* $|A \cup B|$ là số phần tử trong hợp của hai tập (tổng số từ duy nhất)
* Giá trị trả về nằm trong khoảng [0, 1], với 1 là hoàn toàn giống nhau và 0 là hoàn toàn khác nhau

Ưu điểm của tương đồng Jaccard:

* Đơn giản và hiệu quả để so sánh tập hợp
* Phù hợp để đo lường sự chồng lấp của từ vựng
* Không bị ảnh hưởng bởi sự lặp lại của các phần tử
3. **Tương đồng cosine**:
```python
cosine_similarity = dot(vectorizer(expected_output), vectorizer(completion)) / (norm(vectorizer(expected_output)) * norm(vectorizer(completion))) * 100
```

Tương đồng cosine đo lường góc giữa hai vector trong không gian đa chiều, một phương pháp rất hiệu quả để so sánh ngữ nghĩa của văn bản. Công thức toán học:

$$\text{cosine}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Trong đó:

* $\vec{A}$ và $\vec{B}$ là các vector nhúng đại diện cho hai đoạn văn bản
* $\vec{A} \cdot \vec{B}$ là tích vô hướng (dot product) của hai vector
* $||\vec{A}||$ và $||\vec{B}||$ là độ dài Euclidean của mỗi vector
* Giá trị trả về nằm trong khoảng [-1, 1], với 1 là hoàn toàn giống nhau, 0 là không liên quan, và -1 là hoàn toàn trái ngược
* Vector nhúng được tạo bằng FastEmbedEmbeddings, biến đổi văn bản thành vector 384 chiều

Ưu điểm của tương đồng cosine:

* Bỏ qua độ dài văn bản, chỉ tập trung vào hướng ngữ nghĩa
* Hiệu quả đối với dữ liệu thưa (sparse data)
* Phản ánh tốt mối quan hệ ngữ nghĩa giữa các văn bản

### Kết hợp các phương pháp đo lường
Hệ thống kết hợp các phương pháp đo lường trên để tạo ra một đánh giá toàn diện hơn về mức độ tương đồng giữa câu hỏi và câu trả lời:
```python
combined_similarity = (
    0.4 * cosine_similarity + 
    0.3 * jaccard_similarity + 
    0.2 * word_similarity + 
    0.1 * string_similarity
)
```
Việc kết hợp nhiều phương pháp giúp:
* Cân bằng giữa tương đồng cấu trúc và tương đồng ngữ nghĩa
* Giảm thiểu sai số của từng phương pháp riêng lẻ
* Cung cấp đánh giá chính xác hơn về mức độ liên quan giữa câu hỏi và câu trả lời

### Cải tiến đánh giá 

1. **Dynamic Evaluation Weight**

Hệ thống tự động điều chỉnh trọng số đánh giá dựa trên loại truy vấn:
```python
# Ví dụ cho truy vấn về cấu trúc dữ liệu
{"structure": 0.5, "content": 0.3, "relevance": 0.2}

# Ví dụ cho truy vấn về triển khai thuật toán
{"structure": 0.2, "content": 0.5, "relevance": 0.3}
```

2. **Phát hiện chủ đề DSA**
Hệ thống sử dụng từ điển chủ đề DSA để phát hiện nội dung câu hỏi và câu trả lời:
```python
DSA_KEYWORDS = {
    "sort": ["sort", "sorting", "order", "arrange", "sequence", "sắp xếp", "xếp", "thứ tự"],
    "search": ["search", "find", "locate", "query", "lookup", "tìm kiếm", "truy vấn", "tìm", "tra cứu"],
    "tree": ["tree", "binary", "node", "leaf", "root", "branch", "cây", "nhị phân", "nút", "lá", "gốc", "nhánh"],
    "graph": ["graph", "vertex", "edge", "node", "path", "network", "đồ thị", "đỉnh", "cạnh", "nút", "đường đi", "mạng"],
    "hash": ["hash", "map", "key", "value", "collision", "bucket", "bảng băm", "khóa", "giá trị", "va chạm", "băm"],
    "queue": ["queue", "enqueue", "dequeue", "fifo", "first in first out", "hàng đợi", "vào trước ra trước"],
    "stack": ["stack", "push", "pop", "lifo", "last in first out", "ngăn xếp", "vào sau ra trước", "đẩy", "lấy"],
    "linked_list": ["linked list", "node", "pointer", "next", "previous", "danh sách liên kết", "con trỏ", "kế tiếp", "trước đó"],
    "array": ["array", "index", "element", "position", "contiguous", "mảng", "phần tử", "chỉ mục", "vị trí"],
    "recursion": ["recursion", "recursive", "base case", "call stack", "đệ quy", "trường hợp cơ sở", "trường hợp cơ bản"],
    "dynamic_programming": ["dynamic programming", "dp", "memoization", "subproblem", "quy hoạch động", "bài toán con", "ghi nhớ"],
    "greedy": ["greedy", "optimal", "choice", "local optimum", "tham lam", "tối ưu cục bộ", "lựa chọn"]
}
```

3. **Xếp hạng chất lượng**
Dựa trên điểm tổng hợp, hệ thống cung cấp xếp hạng từ A đến F:

| Xếp hạng | Điểm số     |
|----------|-------------|
| A        | ≥ 90 điểm   |
| B        | ≥ 80 điểm   |
| C        | ≥ 70 điểm   |
| D        | ≥ 60 điểm   |
| F        | < 60 điểm   |

### Lưu trữ kết quả đánh giá
Hệ thống lưu trữ kết quả đánh giá trong thư mục "evaluations" dưới định dạng JSON, cho phép:
* Phân tích xu hướng chất lượng câu trả lời theo thời gian
* Xác định các lĩnh vực cần cải thiện
* So sánh hiệu suất giữa các phiên bản của hệ thống

Hệ thống đánh giá tự động này giúp DSA Flask không ngừng cải thiện chất lượng câu trả lời và đảm bảo thông tin chính xác, có cấu trúc rõ ràng và liên quan đến câu hỏi của người dùng.

## Phát triển và Mở rộng

Dự án được thiết kế theo cấu trúc module, cho phép dễ dàng mở rộng thêm các chức năng mới hoặc thay đổi các thành phần hiện có.