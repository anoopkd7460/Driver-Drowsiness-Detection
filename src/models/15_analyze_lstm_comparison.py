import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Phase 15 - LSTM vs LSTM + Attention Analysis
BASE_DIR = "."

LSTM_PREDICTIONS = os.path.join(BASE_DIR,"results","lstm_baseline","test_predictions.csv")
ATTENTION_PREDICTIONS = os.path.join(BASE_DIR,"results","lstm_attention","test_predictions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR,"results","phase15_comparison")

os.makedirs(OUTPUT_DIR,exist_ok=True)

print("="*70)
print("PHASE 15 — LSTM vs LSTM + ATTENTION")
print("="*70)

# 1. Load predictions
print("\n"+"="*70)
print("1. LOADING TEST PREDICTIONS")
print("="*70)

lstm = pd.read_csv(LSTM_PREDICTIONS)
attention = pd.read_csv(ATTENTION_PREDICTIONS)

print("\nLSTM columns:")
print(list(lstm.columns))
print("\nAttention columns:")
print(list(attention.columns))
print("\nLSTM shape      :",lstm.shape)
print("Attention shape :",attention.shape)

# 2. Display sample
print("\n"+"="*70)
print("2. SAMPLE PREDICTIONS")
print("="*70)

print("\nLSTM:")
print(lstm.head())

print("\nAttention:")
print(attention.head())

# 3. Identify column names
def find_column(df,candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find any of these columns: {candidates}")


TRUE_COL = find_column(lstm,["label","true_label","y_true","actual"])
LSTM_PRED_COL = find_column(lstm,["prediction","predicted_label","y_pred","pred"])
ATTENTION_PRED_COL = find_column(attention,["prediction","predicted_label","y_pred","pred"])

# 4. Identify probability columns
def find_probability_column(df):
    candidates = [
        "probability_drowsy",
        "probability",
        "prediction_probability",
        "predicted_probability",
        "prob",
        "score",
        "y_prob"
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


LSTM_PROB_COL = find_probability_column(lstm)
ATTENTION_PROB_COL = find_probability_column(attention)

print("\nDetected columns:")
print("True label       :",TRUE_COL)
print("LSTM prediction  :",LSTM_PRED_COL)
print("Attention pred.  :",ATTENTION_PRED_COL)
print("LSTM probability :",LSTM_PROB_COL)
print("Attention prob.  :",ATTENTION_PROB_COL)

# 5. Consistency check
print("\n"+"="*70)
print("3. CONSISTENCY CHECK")
print("="*70)

if len(lstm) != len(attention):
    raise ValueError("LSTM and Attention prediction files have different lengths.")

if not np.array_equal(lstm[TRUE_COL].values,attention[TRUE_COL].values):
    raise ValueError("True labels are not aligned between the two models.")

print("\nTrue labels aligned: YES")

# 6. Model metrics
y_true = lstm[TRUE_COL].astype(int).values
y_lstm = lstm[LSTM_PRED_COL].astype(int).values
y_attention = attention[ATTENTION_PRED_COL].astype(int).values


def calculate_metrics(y_true,y_pred):
    result = {
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,zero_division=0),
        "recall": recall_score(y_true,y_pred,zero_division=0),
        "f1": f1_score(y_true,y_pred,zero_division=0)
    }
    return result


lstm_metrics = calculate_metrics(y_true,y_lstm)
attention_metrics = calculate_metrics(y_true,y_attention)

comparison = pd.DataFrame({
    "LSTM": lstm_metrics,
    "LSTM_Attention": attention_metrics
})

comparison["Difference_Attention_minus_LSTM"] = (
    comparison["LSTM_Attention"]-comparison["LSTM"]
)

print("\nModel comparison:")
print(comparison)

comparison.to_csv(
    os.path.join(OUTPUT_DIR,"model_comparison.csv")
)

# 7. Confusion matrices
print("\n"+"="*70)
print("4. CONFUSION MATRICES")
print("="*70)

cm_lstm = confusion_matrix(y_true,y_lstm)
cm_attention = confusion_matrix(y_true,y_attention)

print("\nLSTM:")
print(cm_lstm)

print("\nLSTM + Attention:")
print(cm_attention)

cm_df = pd.DataFrame({
    "LSTM_TN":[cm_lstm[0,0]],
    "LSTM_FP":[cm_lstm[0,1]],
    "LSTM_FN":[cm_lstm[1,0]],
    "LSTM_TP":[cm_lstm[1,1]],
    "Attention_TN":[cm_attention[0,0]],
    "Attention_FP":[cm_attention[0,1]],
    "Attention_FN":[cm_attention[1,0]],
    "Attention_TP":[cm_attention[1,1]]
})

cm_df.to_csv(
    os.path.join(OUTPUT_DIR,"confusion_matrix_comparison.csv"),
    index=False
)

# 8. Prediction agreement
print("\n"+"="*70)
print("5. PREDICTION AGREEMENT")
print("="*70)

same_prediction = y_lstm == y_attention
agreement_count = np.sum(same_prediction)
agreement_rate = np.mean(same_prediction)

print(f"\nSame predictions : {agreement_count}/{len(y_true)}")
print(f"Agreement rate   : {agreement_rate*100:.2f}%")
print(f"Different predictions : {np.sum(~same_prediction)}")

# 9. Error analysis
print("\n"+"="*70)
print("6. ERROR ANALYSIS")
print("="*70)

lstm_correct = y_lstm == y_true
attention_correct = y_attention == y_true

print("\nLSTM correct predictions      :",np.sum(lstm_correct))
print("Attention correct predictions :",np.sum(attention_correct))
print("\nLSTM errors      :",np.sum(~lstm_correct))
print("Attention errors :",np.sum(~attention_correct))

# 10. Error categories
lstm_fp = np.sum((y_true == 0)&(y_lstm == 1))
lstm_fn = np.sum((y_true == 1)&(y_lstm == 0))

attention_fp = np.sum((y_true == 0)&(y_attention == 1))
attention_fn = np.sum((y_true == 1)&(y_attention == 0))

print("\nLSTM:")
print("  False Positives :",lstm_fp)
print("  False Negatives :",lstm_fn)

print("\nAttention:")
print("  False Positives :",attention_fp)
print("  False Negatives :",attention_fn)

# 11. Different predictions
different_mask = y_lstm != y_attention

if np.sum(different_mask) > 0:
    different_df = pd.DataFrame({
        "index":np.where(different_mask)[0],
        "true_label":y_true[different_mask],
        "lstm_prediction":y_lstm[different_mask],
        "attention_prediction":y_attention[different_mask]
    })
else:
    different_df = pd.DataFrame(
        columns=[
            "index",
            "true_label",
            "lstm_prediction",
            "attention_prediction"
        ]
    )

different_df.to_csv(
    os.path.join(OUTPUT_DIR,"different_predictions.csv"),
    index=False
)

print("\nDifferent prediction records saved.")

# 12. ROC-AUC
print("\n"+"="*70)
print("7. ROC-AUC")
print("="*70)


def calculate_auc(df,probability_column):
    if probability_column is None:
        return None

    probabilities = df[probability_column].astype(float).values
    return roc_auc_score(y_true,probabilities)


lstm_auc = calculate_auc(lstm,LSTM_PROB_COL)
attention_auc = calculate_auc(attention,ATTENTION_PROB_COL)

print("\nLSTM ROC-AUC      :",lstm_auc)
print("Attention ROC-AUC :",attention_auc)

# 13. Classification reports
print("\n"+"="*70)
print("8. CLASSIFICATION REPORTS")
print("="*70)

print("\nLSTM:")
print(classification_report(
        y_true,
        y_lstm,
        target_names=["Not Drowsy","Drowsy"],
        zero_division=0
    ))

print("\nLSTM + Attention:")
print(classification_report(
        y_true,
        y_attention,
        target_names=["Not Drowsy","Drowsy"],
        zero_division=0
    ))

# 14. Save summary
summary = pd.DataFrame([
    {
        "model":"LSTM",
        "accuracy":lstm_metrics["accuracy"],
        "precision":lstm_metrics["precision"],
        "recall":lstm_metrics["recall"],
        "f1":lstm_metrics["f1"],
        "false_positives":lstm_fp,
        "false_negatives":lstm_fn,
        "roc_auc":lstm_auc
    },
    {
        "model":"LSTM_Attention",
        "accuracy":attention_metrics["accuracy"],
        "precision":attention_metrics["precision"],
        "recall":attention_metrics["recall"],
        "f1":attention_metrics["f1"],
        "false_positives":attention_fp,
        "false_negatives":attention_fn,
        "roc_auc":attention_auc
    }
])

summary.to_csv(
    os.path.join(OUTPUT_DIR,"phase15_summary.csv"),
    index=False
)

# 15. Final conclusion
print("\n"+"="*70)
print("PHASE 15 INITIAL CONCLUSION")
print("="*70)

if np.array_equal(y_lstm,y_attention):
    print("\nLSTM and LSTM + Attention produced IDENTICAL test predictions.")
    print("\nAttention provided no measurable improvement on the current held-out test set.")
else:
    print("\nLSTM and Attention produced different predictions.")

print("\nTest set remains unchanged.")
print("Threshold remains unchanged.")
print("No mouth/yawning features are used.")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n"+"="*70)
print("PHASE 15 ANALYSIS COMPLETED")
print("="*70)