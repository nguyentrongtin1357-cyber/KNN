# BÁO CÁO THỰC NGHIỆM CHI TIẾT: PHÂN LOẠI MỨC ĐỘ PHỔ BIẾN CỦA BÀI VIẾT BẰNG THUẬT TOÁN K-NEAREST NEIGHBORS (KNN)
## ĐƯỢC THIẾT KẾ ĐỦ VÀ KHỚP 100% SỐ THỨ TỰ 21 BƯỚC TÀI LIỆU GỐC

---

## 1. XÁC ĐỊNH BÀI TOÁN (PROBLEM DEFINITION)
* **Mục tiêu nghiệp vụ:** Dự đoán một bài báo trực tuyến trên Mashable sẽ trở nên **Phổ biến (Popular)** hay **Ít phổ biến (Not Popular)**.
* **Đầu vào (Features):** 58-60 thuộc tính trích xuất từ bài viết.
* **Đầu ra (Target):** Nhãn nhị phân `popular` $\in {0, 1}$.
* **Phân loại bài toán:** Phân loại nhị phân (Binary Classification).

---

## 2. XÁC ĐỊNH BẢN CHẤT CỦA BÀI TOÁN ML
* **Learning Paradigm:** Học có giám sát (Supervised Learning).
* **Phương pháp gán nhãn:** Sử dụng ngưỡng trung vị Median ($shares = 1,400$) để gán nhãn cân bằng 50:50.

---

## 3. KHẢO SÁT LĨNH VỰC VÀ KHÔNG GIAN DỮ LIỆU (DOMAIN & DATA UNDERSTANDING)
* Bộ dữ liệu UCI Online News Popularity gồm 39,644 mẫu dữ liệu với 6 nhóm thuộc tính: Thống kê văn bản, Đa phương tiện, Từ khóa & Chủ đề LDA, Kênh tin tức, Thời gian xuất bản, Cực tính cảm xúc.

---

## 4. KHÁM PHÁ VÀ XỬ LÝ DỮ LIỆU (DATA EXPLORATION & CLEANING)
* **Missing Values & Duplicates:** 0 ô trống $NaN$, 0 dòng trùng lặp.
* **Outliers:** Biến `shares` có giá trị lớn nhất 843,300. Xử lý triệt tiêuOutliers bằng phương pháp Quantile Normalization.

---

## 5. CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING)
* **Phương pháp:** Sử dụng `QuantileTransformer(output_distribution='normal')` đưa các đặc trưng về phân phối Chuẩn Gaussian $N(0, 1)$ mà không bị ảnh hưởng bởi Outliers.

---

## 6. XỬ LÝ BIẾN PHÂN LOẠI (CATEGORICAL DATA & ENCODING)
* Các biến phân loại như ngày đăng và kênh tin tức đã được mã hóa One-Hot sẵn trong tập dữ liệu UCI gốc.

---

## 7. LỰA CHỌN THUẬT TOÁN VÀ HÀM MẤT MÁT (ALGORITHM & LOSS FUNCTION SELECTION)
* Thuật toán K-Nearest Neighbors (KNN) kết hợp phép đo khoảng cách Manhattan ($L_1$) và trọng số nghịch đảo `distance`.

---

## 8. KỸ THUẬT TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)
* Tạo 2 biến tương tác phái sinh: `kw_img_interaction` và `self_ref_kw_ratio`.
* Sử dụng `SelectKBest(f_classif, k=25)` lọc ra 25 thuộc tính quan trọng nhất.

---

## 9. CHIA DỮ LIỆU VÀ KIỂM SOÁT DATA LEAKAGE (DATA SPLITTING & LEAKAGE PREVENTION)
* Loại bỏ `shares`, `url`, `timedelta`, `popular` khỏi tập $X$ trước khi chia tập dữ liệu.
* Đóng gói toàn bộ các bước tiền xử lý trong `Pipeline`.

---

## 10. LỰA CHỌN PHƯƠNG PHÁP CHIA DỮ LIỆU (DATA SPLITTING STRATEGY)
* Sử dụng `train_test_split(stratify=y, test_size=0.2, random_state=42)`. Tập Train: 31,715 mẫu; Tập Test: 7,929 mẫu.

---

## 11. XÂY DỰNG MÔ HÌNH CƠ SỞ (BASELINE MODEL)
* **Dummy Baseline:** Accuracy: 0.5336 | Precision: 0.5336 | F1: 0.6959 | ROC-AUC: 0.5000.
* **Baseline KNN ($K=5$, Uniform, Euclidean):** Accuracy: 0.6257 | Precision: 0.6493 | F1: 0.6492 | ROC-AUC: 0.6629.

---

## 12. NGUYÊN LÝ NO FREE LUNCH (NO FREE LUNCH THEOREM)
* Không có mô hình nào tối ưu tuyệt đối cho mọi tập dữ liệu, cần thực nghiệm tinh chỉnh siêu tham số hệ thống.

---

## 13. PHÂN TÍCH BIAS VÀ VARIANCE (BIAS–VARIANCE ANALYSIS)
* Baseline KNN có Train Accuracy 0.7503 và Test Accuracy 0.6257 (Chênh lệch: 12.46%).

---

## 14. TỐI ƯU SIÊU THAM SỐ (HYPERPARAMETER TUNING)
* **GridSearchCV:** Tìm ra cấu hình tối ưu $K=35$, `metric='manhattan'`, `weights='distance'`.
* **Threshold Tuning:** Ngưỡng vận hành tối ưu $\tau = 0.48$.

---

## 15. LỰA CHỌN VÀ ĐÁNH GIÁ BẰNG EVALUATION METRICS
* **Accuracy :** 0.6609
* **Precision:** 0.6704
* **Recall   :** 0.7169
* **F1-Score :** 0.6929
* **ROC-AUC  :** 0.7132

---

## 16. KIỂM ĐỊNH CHÉO K-FOLD (K-FOLD CROSS-VALIDATION)
* Kết quả 5-Fold Cross-Validation đạt độ ổn định cao với $STD < 0.005$.

---

## 17. THỰC NGHIỆM HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING & EXPERIMENTATION)
* Thời gian huấn luyện: 0.4084 giây.
* Thời gian dự đoán (7,929 bài test): 6.8180 giây.

---

## 18. KIỂM ĐỊNH THỐNG KÊ ĐỘ TIN CẬY (STATISTICAL SIGNIFICANCE & CONFIDENCE)
* Paired t-test giữa 5-Fold ROC-AUC của Baseline và Best Tuned KNN đạt $p-value = 0.000118 < 0.05$ (Cải thiện có ý nghĩa thống kê rõ ràng).

---

## 19. PHÂN TÍCH LỖI (ERROR ANALYSIS)
* Ma trận nhầm lẫn và phân tích tỷ lệ lỗi theo từng kênh tin tức (Kênh Entertainment & World có tỷ lệ lỗi cao hơn do đặc thù nội dung biến động).

---

## 20. KHẢ NĂNG GIẢI THÍCH MÔ HÌNH (MODEL INTERPRETABILITY)
* Top 3 đặc trưng quan trọng nhất (Permutation Importance): `kw_avg_avg`, `kw_img_interaction`, `data_channel_is_entertainment`.

---

## 21. CHU TRÌNH LẶP CẢI TIẾN MÔ HÌNH (ITERATIVE ML DEVELOPMENT CYCLE)

### Bảng Tổng Hợp 4 Vòng Lặp Thực Nghiệm:

| Vòng Lặp | Mô Tả Thay Đổi | Accuracy | Precision | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Iter 1: Dummy Baseline** | Dự đoán ngẫu nhiên / gán đa số nhãn 1 | 0.5336 | 0.5336 | 0.6959 | 0.5000 |
| **Iter 2: Standard Baseline KNN** | QuantileTransform + KBest(25) + KNN(K=5, Euclidean, Uniform) | 0.6257 | 0.6493 | 0.6492 | 0.6629 |
| **Iter 3: Feature Eng + GridSearch KNN** | Thêm biến tương tác + Best Params (K=35, Manhattan, Distance) | 0.6573 | 0.6768 | 0.6808 | 0.7132 |
| **Iter 4: Threshold Tuning (Best Model)** | Tinh chỉnh ngưỡng quyết định tau = 0.48 | **0.6609** | **0.6704** | **0.6929** | **0.7132** |

---

*Báo cáo được khởi tạo tự động và kiểm tra khớp 100% chuẩn số thứ tự 21 bước Machine Learning.*
