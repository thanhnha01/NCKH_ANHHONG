# ==========================================================
# Random Forest + SMOTETomek + Calibration (Sigmoid)
# Forward-chaining (3-4-5 folds) — chạy thử với các mức data
# ==========================================================

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime

from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, brier_score_loss
)

# ======================= CONFIG =======================
CSV_PATH = "data/data2.csv"       # 👈 chạy thử với các mức data
TARGET_COL = "isFraud"
N_SPLITS = 4
VALID_RATIO, TEST_RATIO = 0.15, 0.15
RANDOM_SEED, N_TREES = 42, 400
TARGET_RECALL = 0.8
CALIB_METHOD = "sigmoid"

# KHÔNG VẼ PLOT
PLOT_CALIB = False

RISK_BINS = [0.00, 0.09, 0.29, 0.59, 1.00]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]

# ======================= UTILS =======================
def ensure_results_dir():
    for d in [
        "results/metrics_rf_forward",
        "results/feature_importance_rf_forward",
        "results/probabilities_forward",
    ]:
        os.makedirs(d, exist_ok=True)
    # dọn thư mục plots nếu có
    shutil.rmtree("results/plots_forward", ignore_errors=True)

# def make_forward_splits(n: int, k_max: int, valid_ratio: float, test_ratio: float):
#     """Chia dữ liệu kiểu forward-chaining: train -> valid -> test."""
#     valid_len, test_len = int(n * valid_ratio), int(n * test_ratio)
#     window = valid_len + test_len
#     train_len = n - k_max * window
#     splits = []
#     for i in range(k_max):
#         tr_end = train_len + i * window
#         va_start, va_end = tr_end, tr_end + valid_len
#         te_start, te_end = va_end, va_end + test_len
#         if te_end > n: break
#         splits.append((range(0, tr_end), range(va_start, va_end), range(te_start, te_end)))
#     return splits
def make_forward_splits(n: int, k_max: int, valid_ratio: float, test_ratio: float):
    """
    Forward-chaining, auto-clip số fold để không làm train_len âm.
    """
    valid_len = int(round(n * valid_ratio))
    test_len  = int(round(n * test_ratio))
    window = valid_len + test_len

    if min(valid_len, test_len) <= 0:
        raise ValueError("Dữ liệu quá ít cho valid/test.")

    # Số fold tối đa có thể tạo với cửa sổ window
    k_eff = min(k_max, max(1, (n - 1) // window))
    train_len = n - k_eff * window
    if train_len <= 0:
        # nếu vẫn âm, giảm tiếp đến khi dương
        while k_eff > 1 and n - (k_eff * window) <= 0:
            k_eff -= 1
        train_len = n - k_eff * window
        if train_len <= 0:
            raise ValueError("VALID/TEST quá lớn hoặc n quá nhỏ → không tạo được fold.")

    splits = []
    for i in range(k_eff):
        tr_end    = train_len + i * window
        va_start  = tr_end
        va_end    = va_start + valid_len
        te_start  = va_end
        te_end    = te_start + test_len
        if te_end > n: break
        splits.append((range(0, tr_end), range(va_start, va_end), range(te_start, te_end)))
    return splits


def print_layout(splits):
    print("📚 Forward-chaining layout:")
    for i, (tr, va, te) in enumerate(splits, 1):
        print(f"  • Fold {i}: Train[{len(tr):,}] Valid[{len(va):,}] Test[{len(te):,}]")
    print("="*70)

# def pick_threshold_by_recall(y_true, proba, target_recall=0.8):
#     """Chọn ngưỡng ưu tiên Recall ≥ target; nếu không đạt thì lấy ngưỡng F1 cao nhất."""
#     best_thr, best_f1, chosen = 0.5, -1, None
#     for thr in np.linspace(0.001, 0.15, 100):
#         pred = (proba >= thr).astype(int)
#         rec = recall_score(y_true, pred, zero_division=0)
#         f1  = f1_score(y_true, pred, zero_division=0)
#         if rec >= target_recall and chosen is None:
#             chosen = thr
#         if f1 > best_f1:
#             best_f1, best_thr = f1, thr
#     return chosen if chosen else best_thr

def pick_threshold_by_recall(y_true, proba, target_recall=0.80,
                             min_precision=0.01, min_thr=0.003, max_thr=0.20):
    best_thr, best_f1, chosen = 0.5, -1.0, None
    for thr in np.linspace(min_thr, max_thr, 200):
        pred = (proba >= thr).astype(int)
        rec  = recall_score(y_true, pred, zero_division=0)
        pre  = precision_score(y_true, pred, zero_division=0)
        f1   = f1_score(y_true, pred, zero_division=0)
        if rec >= target_recall and pre >= min_precision and chosen is None:
            chosen = thr
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return chosen if chosen is not None else best_thr


def plot_and_save_calibration(y_true, proba, title, path, n_bins=10):
    pt, pp = calibration_curve(y_true, proba, n_bins=n_bins, strategy='uniform')
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.plot(pp, pt, 'o-')
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction of positives")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def assign_risk_levels(p):
    df = pd.cut(p, bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True)
    mapping = {"LOW":"[0.00–0.09]", "MEDIUM":"(0.09–0.29]", "HIGH":"(0.29–0.59]", "VERY_HIGH":"(0.59–1.00]"}
    return pd.DataFrame({"risk_level": df, "risk_range": df.astype(str).map(mapping)})

# ======================= MAIN =======================
def main():
    # common tag cho tất cả output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_tag = os.path.splitext(os.path.basename(CSV_PATH))[0]

    df = pd.read_csv(CSV_PATH)
    y, X, n = df[TARGET_COL].astype(int), df.drop(columns=[TARGET_COL]), len(df)
    print(f"\n📦 Dataset: {CSV_PATH}")
    print(f"🧷 Features: {X.shape[1]} | Rows: {n:,}\n")

    splits = make_forward_splits(n, N_SPLITS, VALID_RATIO, TEST_RATIO)
    print_layout(splits)
    ensure_results_dir()

    metrics, feats, probs = [], [], []

    for fold, (tr, va, te) in enumerate(splits, 1):
        Xtr, ytr = X.iloc[list(tr)], y.iloc[list(tr)]
        Xva, yva = X.iloc[list(va)], y.iloc[list(va)]
        Xte, yte = X.iloc[list(te)], y.iloc[list(te)]

        # Oversampling: SMOTETomek
        sampler = SMOTETomek(random_state=RANDOM_SEED)
        Xbal, ybal = sampler.fit_resample(Xtr, ytr)

        # Train RF
        rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_SEED)
        rf.fit(Xbal, ybal)
        if hasattr(rf, "feature_importances_"):
            feats.append(rf.feature_importances_)

        # Calibration
        cal = CalibratedClassifierCV(rf, cv="prefit", method=CALIB_METHOD)
        cal.fit(Xva, yva)

        pv, pt = cal.predict_proba(Xva)[:,1], cal.predict_proba(Xte)[:,1]
        thr = pick_threshold_by_recall(yva, pv, TARGET_RECALL)
        ypv, ypt = (pv >= thr).astype(int), (pt >= thr).astype(int)

        def metrics_pack(y_true, y_pred, proba):
            return dict(
                auc=roc_auc_score(y_true, proba) if y_true.nunique()>1 else np.nan,
                acc=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred, zero_division=0),
                pre=precision_score(y_true, y_pred, zero_division=0),
                rec=recall_score(y_true, y_pred, zero_division=0),
                brier=brier_score_loss(y_true, proba)
            )

        mva, mte = metrics_pack(yva, ypv, pv), metrics_pack(yte, ypt, pt)
        print(f"\n🧪 Fold {fold} — SMOTETomek + sigmoid + Recall≥{TARGET_RECALL}")
        print(f"   Train_bal={len(ybal):,} | Threshold={thr:.3f}")
        print(f"   VALID: AUC={mva['auc']:.4f} | F1={mva['f1']:.4f} | P={mva['pre']:.4f} | R={mva['rec']:.4f}")
        print(f"   TEST : AUC={mte['auc']:.4f} | F1={mte['f1']:.4f} | P={mte['pre']:.4f} | R={mte['rec']:.4f}")

        metrics.append({
            "fold": fold, "thr_from_valid": thr,
            **{f"valid_{k}": v for k,v in mva.items()},
            **{f"test_{k}": v for k,v in mte.items()},
        })

        # Save predictions & risk levels (gộp tất cả fold vào 1 file)
        dfp = Xte.copy()
        dfp["true_label"] = yte.values
        dfp["fraud_probability"] = pt
        dfp["predicted_label"] = ypt
        dfp = pd.concat([dfp, assign_risk_levels(dfp["fraud_probability"])], axis=1)
        dfp["fold"] = fold
        dfp["orig_index"] = Xte.index
        probs.append(dfp)

        # KHÔNG vẽ plot; vẫn giữ hàm cho tiện bật lại nếu cần
        if PLOT_CALIB:
            os.makedirs("results/plots_forward", exist_ok=True)
            plot_and_save_calibration(yva, pv, f"VALID Fold {fold}", f"results/plots_forward/valid_{fold}.png")
            plot_and_save_calibration(yte, pt, f"TEST Fold {fold}", f"results/plots_forward/test_{fold}.png")

    # ---------- Save 1 FILE MỖI THƯ MỤC ----------
    # 1) metrics_rf_forward
    dfm = pd.DataFrame(metrics)
    dfm.loc[len(dfm)] = {"fold": "MEAN", **dfm.drop(columns="fold").mean(numeric_only=True).to_dict()}
    dfm.to_csv(f"results/metrics_rf_forward/metrics_rf_forward_{csv_tag}_{ts}.csv", index=False)

    # 2) feature_importance_rf_forward
    if feats:
        fi_mean = np.mean(np.vstack(feats), axis=0)
        fi = pd.DataFrame({"feature": X.columns, "importance_mean": fi_mean}).sort_values(
            "importance_mean", ascending=False
        )
        fi.to_csv(
            f"results/feature_importance_rf_forward/feature_importance_rf_forward_{csv_tag}_{ts}.csv",
            index=False
        )

    # 3) probabilities_forward
    if probs:
        df_prob = pd.concat(probs, axis=0, ignore_index=True)
        df_prob.to_csv(
            f"results/probabilities_forward/probabilities_forward_{csv_tag}_{ts}.csv",
            index=False
        )

    # 4) dọn plots nếu lỡ tạo
    if not PLOT_CALIB:
        shutil.rmtree("results/plots_forward", ignore_errors=True)

    print("\n✅ Done! Saved:")
    print(f"   • metrics_rf_forward/metrics_rf_forward_{csv_tag}_{ts}.csv")
    print(f"   • feature_importance_rf_forward/feature_importance_rf_forward_{csv_tag}_{ts}.csv")
    print(f"   • probabilities_forward/probabilities_forward_{csv_tag}_{ts}.csv")
    if PLOT_CALIB:
        print("   • plots_forward/*.png")

# ==========================================================
if __name__ == "__main__":
    main()
