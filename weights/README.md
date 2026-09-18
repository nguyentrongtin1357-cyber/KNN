# THƯ MỤC WEIGHTS VÀ THAM SỐ TỐI ƯU MÔ HÌNH KNN

Thư mục này lưu trữ toàn bộ **Model Weights**, **Siêu tham số tối ưu (Hyperparameters)** và **Danh sách đặc trưng quan trọng** được trích xuất tự động sau quá trình huấn luyện và tinh chỉnh của dự án phân loại bài viết **UCI Online News Popularity bằng KNN**.

---

## 📂 Danh Sách Các File Trong Thư Mục Weights:

1. **`knn_best_model.joblib`**
   - File nhị phân lưu trữ toàn bộ Pipeline mô hình tối ưu đã fit (`pipe_best`).
   - Đóng gói đầy đủ 3 bước: `QuantileTransformer(normal)` $\rightarrow$ `SelectKBest(k=25)` $\rightarrow$ `KNeighborsClassifier(K=35, Manhattan, Distance)`.
   - Được nạp dễ dàng bằng hàm `joblib.load('weights/knn_best_model.joblib')`.

2. **`best_params.json`**
   - File JSON dạng văn bản lưu trữ toàn bộ các siêu tham số tối ưu và kết quả đánh giá thực nghiệm:
     - `n_neighbors`: **35**
     - `metric`: **manhattan** ($L_1$)
     - `weights`: **distance**
     - `decision_threshold`: **0.48**
     - `n_features_selected`: **25**
     - `evaluation_metrics`: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

3. **`selected_features.json`**
   - File JSON lưu danh sách tên 25 thuộc tính quan trọng nhất được thuật toán `SelectKBest` chọn lọc từ 60 thuộc tính ban đầu.

---

## 🚀 Cách Nạp Model Weights Để Dự Đoán (Python):

```python
import joblib
import json
import pandas as pd

# 1. Nạp Siêu Tham Số Tối Ưu
with open('weights/best_params.json', 'r', encoding='utf-8') as f:
    params = json.load(f)

# 2. Nạp Model Weights
model = joblib.load('weights/knn_best_model.joblib')

# 3. Dự đoán trên dữ liệu mới X_new
probs = model.predict_proba(X_new)[:, 1]
predictions = (probs >= params['decision_threshold']).astype(int)
print(predictions)
```

Hoặc chạy trực tiếp file demo dự đoán:
```bash
py predict_with_weights.py
```
