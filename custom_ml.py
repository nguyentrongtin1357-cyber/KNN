import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist
import math
import time
import copy
import joblib

# ---------------------------------------------------------
# 1. Custom Preprocessing & Feature Selection
# ---------------------------------------------------------

class CustomQuantileTransformer:
    BOUNDS_THRESHOLD = 1e-7

    def __init__(self, n_quantiles=1000, subsample=100000, output_distribution='normal', random_state=42):
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state
        self.quantiles_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        if n_samples > self.subsample:
            rng = np.random.RandomState(self.random_state)
            sub_idx = rng.choice(n_samples, size=self.subsample, replace=False)
            X_sub = X_arr[sub_idx]
        else:
            X_sub = X_arr

        n_q = min(len(X_sub), self.n_quantiles)
        percentiles = np.linspace(0, 100, n_q)
        quantiles = []
        for j in range(n_features):
            col_quantiles = np.percentile(X_sub[:, j], percentiles)
            quantiles.append(col_quantiles)
        self.quantiles_ = np.column_stack(quantiles)
        self.percentiles_ = percentiles / 100.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        X_trans = np.zeros_like(X_arr)
        
        for j in range(n_features):
            # Interpolate rank
            ranks = np.interp(X_arr[:, j], self.quantiles_[:, j], self.percentiles_)
            # Clip ranks to avoid inf in norm.ppf using BOUNDS_THRESHOLD = 1e-7
            ranks = np.clip(ranks, self.BOUNDS_THRESHOLD, 1.0 - self.BOUNDS_THRESHOLD)
            if self.output_distribution == 'normal':
                X_trans[:, j] = stats.norm.ppf(ranks)
            else:
                X_trans[:, j] = ranks
        return X_trans

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def custom_f_classif(X, y):
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int32)
    
    classes = np.unique(y_arr)
    n_samples, n_features = X_arr.shape
    
    f_scores = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    grand_mean = np.mean(X_arr, axis=0)
    
    ss_between = np.zeros(n_features)
    ss_within = np.zeros(n_features)
    
    for c in classes:
        mask = (y_arr == c)
        n_c = np.sum(mask)
        if n_c > 0:
            mean_c = np.mean(X_arr[mask], axis=0)
            ss_between += n_c * (mean_c - grand_mean) ** 2
            ss_within += np.sum((X_arr[mask] - mean_c) ** 2, axis=0)
            
    df_between = len(classes) - 1
    df_within = n_samples - len(classes)
    
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    ms_within[ms_within == 0] = 1e-12
    
    f_scores = ms_between / ms_within
    p_values = 1.0 - stats.f.cdf(f_scores, df_between, df_within)
    return f_scores, p_values


class CustomSelectKBest:
    def __init__(self, score_func=custom_f_classif, k=25):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.selected_indices_ = None

    def fit(self, X, y):
        scores, _ = self.score_func(X, y)
        scores = np.nan_to_num(scores, nan=0.0)
        self.scores_ = scores
        if isinstance(self.k, int):
            # Stable tie-break sorting matching scikit-learn
            self.selected_indices_ = np.argsort(-scores, kind='mergesort')[:self.k]
        else:
            self.selected_indices_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.selected_indices_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_trans = X
        for name, step in self.steps[:-1]:
            X_trans = step.fit_transform(X_trans, y)
        self.steps[-1][1].fit(X_trans, y)
        return self

    def transform(self, X):
        X_trans = X
        for name, step in self.steps[:-1]:
            X_trans = step.transform(X_trans)
        return X_trans

    def predict(self, X, threshold=0.5):
        X_trans = self.transform(X)
        estimator = self.steps[-1][1]
        try:
            return estimator.predict(X_trans, threshold=threshold)
        except TypeError:
            return estimator.predict(X_trans)

    def predict_proba(self, X):
        X_trans = self.transform(X)
        return self.steps[-1][1].predict_proba(X_trans)


# ---------------------------------------------------------
# 2. Custom Data Splitting & Cross Validation
# ---------------------------------------------------------

def custom_train_test_split(X, y, test_size=0.2, stratify=None, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = len(y)
    
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_arr = X.values
        X_is_df = True
    else:
        X_arr = np.asarray(X)
        X_is_df = False
        
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y_arr = y.values.ravel()
        y_is_series = True
    else:
        y_arr = np.asarray(y).ravel()
        y_is_series = False

    if stratify is not None:
        strat_y = np.asarray(stratify).ravel()
        classes, counts = np.unique(strat_y, return_counts=True)
        train_idx = []
        test_idx = []
        for c in classes:
            c_idx = np.where(strat_y == c)[0]
            rng.shuffle(c_idx)
            n_test_c = int(np.round(len(c_idx) * test_size))
            test_idx.extend(c_idx[:n_test_c])
            train_idx.extend(c_idx[n_test_c:])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_test = int(np.round(n_samples * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    if X_is_df and isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        
    if y_is_series and isinstance(y, pd.Series):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    return X_train, X_test, y_train, y_test


class CustomStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        rng = np.random.RandomState(self.random_state)
        y_arr = np.asarray(y).ravel()
        classes = np.unique(y_arr)
        
        folds = [[] for _ in range(self.n_splits)]
        
        for c in classes:
            c_idx = np.where(y_arr == c)[0]
            if self.shuffle:
                rng.shuffle(c_idx)
            
            # Split c_idx into n_splits parts
            split_parts = np.array_split(c_idx, self.n_splits)
            for fold_i in range(self.n_splits):
                folds[fold_i].extend(split_parts[fold_i])
                
        n_samples = len(y_arr)
        all_indices = np.arange(n_samples)
        
        for fold_i in range(self.n_splits):
            test_idx = np.array(folds[fold_i])
            train_idx = np.setdiff1d(all_indices, test_idx)
            if self.shuffle:
                rng.shuffle(train_idx)
                rng.shuffle(test_idx)
            yield train_idx, test_idx


def _evaluate_single_fold(estimator, X_tr, X_te, y_tr, y_te):
    est = copy.deepcopy(estimator)
    est.fit(X_tr, y_tr)
    y_pred = est.predict(X_te)
    
    acc = custom_accuracy_score(y_te, y_pred)
    f1 = custom_f1_score(y_te, y_pred)
    
    if hasattr(est, 'predict_proba'):
        y_prob = est.predict_proba(X_te)[:, 1]
        auc = custom_roc_auc_score(y_te, y_prob)
    else:
        auc = np.nan
        
    return acc, f1, auc


def custom_cross_validate(estimator, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1):
    if isinstance(cv, int):
        cv_gen = CustomStratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_gen = cv

    splits = list(cv_gen.split(X, y))
    
    def get_fold_data(train_idx, test_idx):
        if isinstance(X, pd.DataFrame):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_tr, X_te = X[train_idx], X[test_idx]
            
        if isinstance(y, pd.Series):
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_tr, y_te = y[train_idx], y[test_idx]
        return X_tr, X_te, y_tr, y_te

    if n_jobs != 1:
        results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib.delayed(_evaluate_single_fold)(estimator, *get_fold_data(tr_idx, te_idx))
            for tr_idx, te_idx in splits
        )
    else:
        results = [
            _evaluate_single_fold(estimator, *get_fold_data(tr_idx, te_idx))
            for tr_idx, te_idx in splits
        ]

    test_accs = [r[0] for r in results]
    test_f1s = [r[1] for r in results]
    test_aucs = [r[2] for r in results]

    return {
        'test_accuracy': np.array(test_accs),
        'test_f1': np.array(test_f1s),
        'test_roc_auc': np.array(test_aucs)
    }


# ---------------------------------------------------------
# 3. Custom Evaluation Metrics
# ---------------------------------------------------------

def custom_accuracy_score(y_true, y_pred):
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    return np.mean(y_t == y_p)


def custom_confusion_matrix(y_true, y_pred):
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    tn = np.sum((y_t == 0) & (y_p == 0))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))
    tp = np.sum((y_t == 1) & (y_p == 1))
    return np.array([[tn, fp], [fn, tp]])


def custom_precision_score(y_true, y_pred, zero_division=0):
    cm = custom_confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    if (tp + fp) == 0:
        return zero_division
    return tp / (tp + fp)


def custom_recall_score(y_true, y_pred, zero_division=0):
    cm = custom_confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    if (tp + fn) == 0:
        return zero_division
    return tp / (tp + fn)


def custom_f1_score(y_true, y_pred, zero_division=0):
    p = custom_precision_score(y_true, y_pred, zero_division=zero_division)
    r = custom_recall_score(y_true, y_pred, zero_division=zero_division)
    if (p + r) == 0:
        return zero_division
    return 2 * p * r / (p + r)


def custom_roc_curve(y_true, y_score):
    y_t = np.asarray(y_true).ravel()
    y_s = np.asarray(y_score).ravel()
    
    desc_indices = np.argsort(y_s)[::-1]
    y_t_sorted = y_t[desc_indices]
    y_s_sorted = y_s[desc_indices]
    
    distinct_value_indices = np.where(np.diff(y_s_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, len(y_s_sorted) - 1]
    
    tps = np.cumsum(y_t_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps
    
    n_positives = tps[-1]
    n_negatives = fps[-1]
    
    if n_positives == 0 or n_negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
        
    tpr = tps / n_positives
    fpr = fps / n_negatives
    
    # Prepend (0, 0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[y_s_sorted[0] + 1.0, y_s_sorted[threshold_indices]]
    return fpr, tpr, thresholds


def custom_auc(x, y):
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def custom_roc_auc_score(y_true, y_score):
    fpr, tpr, _ = custom_roc_curve(y_true, y_score)
    return custom_auc(fpr, tpr)


def custom_permutation_importance(estimator, X, y, n_repeats=5, random_state=42, scoring='roc_auc', n_jobs=1):
    rng = np.random.RandomState(random_state)
    
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.copy()
        col_names = X.columns
    else:
        X_arr = np.asarray(X).copy()
        col_names = None
        
    y_arr = np.asarray(y).ravel()
    
    # Baseline score
    if hasattr(estimator, 'predict_proba'):
        baseline_prob = estimator.predict_proba(X_arr)[:, 1]
        baseline_score = custom_roc_auc_score(y_arr, baseline_prob)
    else:
        baseline_pred = estimator.predict(X_arr)
        baseline_score = custom_accuracy_score(y_arr, baseline_pred)

    n_samples, n_features = X_arr.shape
    importances = np.zeros((n_features, n_repeats))

    for j in range(n_features):
        col_backup = X_arr[:, j].copy()
        for r in range(n_repeats):
            permuted_col = rng.permutation(col_backup)
            X_arr[:, j] = permuted_col
            if hasattr(estimator, 'predict_proba'):
                prob = estimator.predict_proba(X_arr)[:, 1]
                score = custom_roc_auc_score(y_arr, prob)
            else:
                pred = estimator.predict(X_arr)
                score = custom_accuracy_score(y_arr, pred)
            importances[j, r] = baseline_score - score
        X_arr[:, j] = col_backup

    class PermResult:
        def __init__(self, imp):
            self.importances_mean = np.mean(imp, axis=1)
            self.importances_std = np.std(imp, axis=1)
            self.importances = imp

    return PermResult(importances)


# ---------------------------------------------------------
# 4. Custom Machine Learning Models
# ---------------------------------------------------------

class CustomKNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict_proba(self, X):
        X_test_arr = np.asarray(X, dtype=np.float32)
        n_test = len(X_test_arr)
        n_train = self.X_train_.shape[0]

        # Guard: k không được vượt quá số mẫu train
        k = min(self.n_neighbors, n_train)

        probs = np.zeros((n_test, len(self.classes_)), dtype=np.float64)
        metric_code = 'cityblock' if self.metric in ['manhattan', 'l1'] else 'euclidean'

        batch_size = 500
        for start_i in range(0, n_test, batch_size):
            end_i = min(start_i + batch_size, n_test)
            X_batch = X_test_arr[start_i:end_i]

            dists = cdist(X_batch, self.X_train_, metric=metric_code)

            # argpartition không ổn định khi có ties -> chỉ dùng để lấy top-k thô
            knn_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]

            for i_local in range(end_i - start_i):
                idx = knn_idx[i_local]
                d = dists[i_local, idx]

                # TIE-BREAK ỔN ĐỊNH: sort theo (distance, original_index)
                # np.lexsort ưu tiên key cuối cùng làm khóa chính -> đặt d sau idx
                order = np.lexsort((idx, d))
                idx = idx[order]
                d_k = d[order]
                y_k = self.y_train_[idx]

                if self.weights == 'distance':
                    w_k = 1.0 / (d_k + 1e-10)
                    prob_1 = np.sum(w_k[y_k == 1]) / np.sum(w_k)
                else:
                    prob_1 = np.mean(y_k == 1)

                probs[start_i + i_local, 1] = prob_1
                probs[start_i + i_local, 0] = 1.0 - prob_1

        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)



