import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def load_model_and_predict(data_input=None):
    """
    Hàm nạp Model Weights và Siêu Tham Số tối ưu đã lưu trong weights/ để dự đoán.
    """
    model_path = 'weights/knn_best_model.joblib'
    params_path = 'weights/best_params.json'
    features_path = 'weights/selected_features.json'
    dt_weights_path = 'weights/detailed_feature_weights.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(params_path):
        raise FileNotFoundError("Chưa tìm thấy file weights! Vui lòng chạy py main.py để huấn luyện và lưu weights.")
    
    # 1. Nạp Siêu tham số tối ưu
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    # 2. Nạp Model Weights (Trained Pipeline)
    model = joblib.load(model_path)
    
    print("================================================================================")
    print("  ĐÃ NẠP THÀNH CÔNG MODEL WEIGHTS VÀ THAM SỐ TỐI ƯU DỰ ÁN KNN")
    print("================================================================================")
    print(f"Thuật toán         : {params['model_name']}")
    print(f"Tham số K (k_nn)   : {params['n_neighbors']}")
    print(f"Phép đo khoảng cách: {params['metric']}")
    print(f"Trọng số           : {params['weights']}")
    print(f"Ngưỡng Tau (Threshold): {params['decision_threshold']}")
    print(f"Số đặc trưng chọn  : {params['n_features_selected']}")
    print(f"ROC-AUC Test Set   : {params['evaluation_metrics']['roc_auc']}")
    print("--------------------------------------------------------------------------------")

    # 3. Hiển thị Top 5 Trọng số đặc trưng chi tiết từ weights/detailed_feature_weights.csv nếu có
    if os.path.exists(dt_weights_path):
        dt_df = pd.read_csv(dt_weights_path)
        print("TOP 5 TRỌNG SỐ ĐẶC TRƯNG QUAN TRỌNG NHẤT (Trích từ weights/detailed_feature_weights.csv):")
        print(dt_df.head(5)[['Rank', 'Feature', 'ANOVA_F_Score', 'Permutation_Importance_Mean']].to_string(index=False))
        print("--------------------------------------------------------------------------------")

    # 4. Thực hiện dự đoán trên dữ liệu đầu vào
    if data_input is None:
        df = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')
        df.columns = [c.strip() for c in df.columns]
        
        # BƯỚC 8: Feature Engineering (nếu chưa có)
        if 'kw_img_interaction' not in df.columns:
            df['kw_img_interaction'] = df['kw_avg_avg'] * df['num_imgs']
        if 'self_ref_kw_ratio' not in df.columns:
            col_ref = 'self_reference_avg_sharess' if 'self_reference_avg_sharess' in df.columns else 'self_reference_avg_shares'
            df['self_ref_kw_ratio'] = df[col_ref] / (df['kw_avg_avg'] + 1e-5)
            
        drop_cols = ['url', 'timedelta', 'shares', 'popular']
        if 'popular' not in df.columns:
            df['popular'] = (df['shares'] >= 1400).astype(int)
        
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df['popular']
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        data_input = X_test.iloc[:10]
        y_true = y_test.iloc[:10].values

    # Tính xác suất và gán nhãn theo ngưỡng tối ưu tau = 0.48
    probs = model.predict_proba(data_input)[:, 1]
    preds = (probs >= params['decision_threshold']).astype(int)
    
    results_df = pd.DataFrame({
        'Sample_ID': range(1, len(preds) + 1),
        'Probability_Popular': np.round(probs, 4),
        'Predicted_Class': preds,
        'Predicted_Label': ['Popular (1)' if p == 1 else 'Not Popular (0)' for p in preds]
    })
    
    if 'y_true' in locals():
        results_df['True_Class'] = y_true
        results_df['Correct'] = results_df['Predicted_Class'] == results_df['True_Class']

    print("\nKẾT QUẢ DỰ ĐOÁN MẪU BẰNG MODEL WEIGHTS ĐÃ LƯU:")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    load_model_and_predict()
