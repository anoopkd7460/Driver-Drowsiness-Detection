from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_METRICS_FILE = PROJECT_ROOT / "results" / "model_evaluation" / "test_metrics.csv"
ROBUSTNESS_FILE = PROJECT_ROOT / "results" / "generalization" / "aggregate_statistics.csv"
REPEATED_RESULTS_FILE = PROJECT_ROOT / "results" / "generalization" / "repeated_video_level_results.csv"
DIFFICULT_VIDEOS_FILE = PROJECT_ROOT / "results" / "generalization" / "video_wise_results.csv"
FINAL_EVALUATION_FILE = PROJECT_ROOT / "results" / "final_evaluation" / "final_test_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "publication_tables"

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)


# Helper functions
def find_column(df,possible_names):
    normalized = {
        str(col).strip().lower():col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def format_percent(value):
    return f"{value*100:.2f}%"


# Start
print("="*75)
print("PUBLICATION-READY RESULTS TABLE GENERATION")
print("="*75)

# 1. Model comparison table
print("\nLoading model evaluation results...")

test_df = pd.read_csv(TEST_METRICS_FILE)

print("\nTest metrics:")
print(test_df)

# Expected structure: model, accuracy, precision, recall, f1, roc_auc
model_column = find_column(test_df,["model"])

if model_column is None:
    raise ValueError("Could not find model column in test_metrics.csv")

required_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

for metric in required_metrics:
    if find_column(test_df,[metric]) is None:
        raise ValueError(f"Could not find '{metric}' in test_metrics.csv")

model_table = test_df[
    [
        model_column,
        find_column(test_df,["accuracy"]),
        find_column(test_df,["precision"]),
        find_column(test_df,["recall"]),
        find_column(test_df,["f1"]),
        find_column(test_df,["roc_auc"])
    ]
].copy()

model_table.columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

# Convert to percentage
model_table_percent = model_table.copy()

for column in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]:
    model_table_percent[column] = (
        model_table_percent[column]*100
    ).round(2)

# Add interpretation
interpretation = []

for model in model_table_percent["Model"]:
    if model == "LSTM_Baseline":
        interpretation.append("Selected final model")
    elif model == "LSTM_Attention":
        interpretation.append("No measurable improvement over baseline")
    elif model == "LSTM_GRL":
        interpretation.append("Lower precision/F1 despite 100% recall")
    else:
        interpretation.append("Comparison model")

model_table_percent["Interpretation"] = interpretation

model_table_path = OUTPUT_DIR / "model_comparison_table.csv"

model_table_percent.to_csv(
    model_table_path,
    index=False
)

print("\nModel Comparison Table:")
print(model_table_percent.to_string(index=False))
print(f"\nSaved: {model_table_path}")

# 2. Confusion matrix table
print("\nCreating confusion matrix table...")

final_metrics = pd.read_csv(FINAL_EVALUATION_FILE)
cm_metrics = {}

for _,row in final_metrics.iterrows():
    metric = str(row["metric"]).strip()
    cm_metrics[metric] = row["value"]

confusion_table = pd.DataFrame({
    "Actual / Predicted":["Not Drowsy","Drowsy"],
    "Not Drowsy":[
        int(cm_metrics["True Negatives"]),
        int(cm_metrics["False Negatives"])
    ],
    "Drowsy":[
        int(cm_metrics["False Positives"]),
        int(cm_metrics["True Positives"])
    ]
})

confusion_table_path = OUTPUT_DIR / "confusion_matrix_table.csv"

confusion_table.to_csv(
    confusion_table_path,
    index=False
)

print("\nConfusion Matrix Table:")
print(confusion_table.to_string(index=False))
print(f"\nSaved: {confusion_table_path}")

# 3. Robustness table
print("\nLoading repeated video-level results...")

repeated_df = pd.read_csv(REPEATED_RESULTS_FILE)

print(f"Repeated experiments: {len(repeated_df)}")

robustness_rows = []

metric_mapping = {
    "Accuracy":"accuracy",
    "Precision":"precision",
    "Recall":"recall",
    "F1-Score":"f1"
}

for display_name,column_name in metric_mapping.items():
    values = repeated_df[column_name].astype(float)

    robustness_rows.append({
        "Metric":display_name,
        "Mean":values.mean()*100,
        "Std":values.std(ddof=1)*100,
        "Minimum":values.min()*100,
        "Maximum":values.max()*100
    })

robustness_table = pd.DataFrame(robustness_rows)

for column in ["Mean","Std","Minimum","Maximum"]:
    robustness_table[column] = robustness_table[column].round(2)

robustness_table_path = OUTPUT_DIR / "video_level_robustness_table.csv"

robustness_table.to_csv(
    robustness_table_path,
    index=False
)

print("\nRepeated Video-Level Robustness:")
print(robustness_table.to_string(index=False))
print(f"\nSaved: {robustness_table_path}")

# 4. Individual repeated experiment table
print("\nCreating repeated experiment table...")

repeated_publication = repeated_df.copy()

rename_map = {
    "seed":"Random Seed",
    "train_videos":"Training Videos",
    "validation_videos":"Validation Videos",
    "train_sequences":"Training Sequences",
    "validation_sequences":"Validation Sequences",
    "train_drowsy":"Training Drowsy Sequences",
    "validation_drowsy":"Validation Drowsy Sequences",
    "best_epoch":"Best Epoch",
    "accuracy":"Accuracy",
    "precision":"Precision",
    "recall":"Recall",
    "f1":"F1-Score",
    "tn":"TN",
    "fp":"FP",
    "fn":"FN",
    "tp":"TP"
}

repeated_publication = repeated_publication.rename(columns=rename_map)

for column in ["Accuracy","Precision","Recall","F1-Score"]:
    if column in repeated_publication.columns:
        repeated_publication[column] = (
            repeated_publication[column]*100
        ).round(2)

repeated_table_path = OUTPUT_DIR / "repeated_video_level_results_table.csv"

repeated_publication.to_csv(
    repeated_table_path,
    index=False
)

print(repeated_publication.to_string(index=False))
print(f"\nSaved: {repeated_table_path}")

# 5. Difficult video table
print("\nCreating difficult-video table...")

video_df = pd.read_csv(DIFFICULT_VIDEOS_FILE)

# Sort by accuracy
video_df = video_df.sort_values(
    by="accuracy",
    ascending=True
)

difficult_table = video_df[
    [
        "seed",
        "video_name",
        "sequences",
        "drowsy_sequences",
        "accuracy",
        "recall_drowsy"
    ]
].copy()

difficult_table.columns = [
    "Seed",
    "Video",
    "Sequences",
    "Drowsy Sequences",
    "Accuracy",
    "Drowsy Recall"
]

difficult_table["Accuracy"] = (
    difficult_table["Accuracy"]*100
).round(2)

difficult_table["Drowsy Recall"] = (
    difficult_table["Drowsy Recall"]*100
).round(2)

difficult_table_path = OUTPUT_DIR / "video_wise_robustness_table.csv"

difficult_table.to_csv(
    difficult_table_path,
    index=False
)

print(difficult_table.to_string(index=False))
print(f"\nSaved: {difficult_table_path}")

# 6. Final research summary table
print("\nCreating final research summary...")

baseline_row = model_table[
    model_table["Model"] == "LSTM_Baseline"
].iloc[0]

summary_table = pd.DataFrame({
    "Category":[
        "Final Model",
        "Input Features",
        "Sequence Length",
        "Frame Rate",
        "Temporal Duration",
        "Official Test Sequences",
        "Official Test Accuracy",
        "Official Test Precision",
        "Official Test Recall",
        "Official Test F1-Score",
        "Official Test ROC-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "Repeated Video-Level Accuracy",
        "Repeated Video-Level Recall",
        "Repeated Video-Level F1"
    ],
    "Result":[
        "LSTM Baseline",
        "EAR Mean + Pitch + Yaw + Roll",
        "30 frames",
        "25 FPS",
        "1.2 seconds",
        "179",
        format_percent(float(baseline_row["Accuracy"])),
        format_percent(float(baseline_row["Precision"])),
        format_percent(float(baseline_row["Recall"])),
        format_percent(float(baseline_row["F1-Score"])),
        format_percent(float(baseline_row["ROC-AUC"])),
        str(int(cm_metrics["True Negatives"])),
        str(int(cm_metrics["False Positives"])),
        str(int(cm_metrics["False Negatives"])),
        str(int(cm_metrics["True Positives"])),
        f"{repeated_df['accuracy'].mean()*100:.2f}% ± "
        f"{repeated_df['accuracy'].std(ddof=1)*100:.2f}%",
        f"{repeated_df['recall'].mean()*100:.2f}% ± "
        f"{repeated_df['recall'].std(ddof=1)*100:.2f}%",
        f"{repeated_df['f1'].mean()*100:.2f}% ± "
        f"{repeated_df['f1'].std(ddof=1)*100:.2f}%"
    ]
})

summary_path = OUTPUT_DIR / "final_research_summary_table.csv"

summary_table.to_csv(
    summary_path,
    index=False
)

print(summary_table.to_string(index=False))
print(f"\nSaved: {summary_path}")

# 7. LaTeX table
print("\nCreating LaTeX model comparison table...")


def latex_escape(text):
    text = str(text)

    replacements = {
        "&":r"\&",
        "%":r"\%",
        "_":r"\_",
        "#":r"\#"
    }

    for old,new in replacements.items():
        text = text.replace(old,new)

    return text


latex_lines = []

latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Comparison of temporal drowsiness detection models.}")
latex_lines.append(r"\label{tab:model_comparison}")
latex_lines.append(r"\begin{tabular}{lccccc}")
latex_lines.append(r"\hline")
latex_lines.append(r"Model & Accuracy & Precision & Recall & F1-Score & ROC-AUC \\")
latex_lines.append(r"\hline")

for _,row in model_table_percent.iterrows():
    model_name = latex_escape(row["Model"])
    accuracy = f'{row["Accuracy"]:.2f}\\%'
    precision = f'{row["Precision"]:.2f}\\%'
    recall = f'{row["Recall"]:.2f}\\%'
    f1 = f'{row["F1-Score"]:.2f}\\%'
    roc_auc_value = f'{row["ROC-AUC"]:.2f}\\%'

    latex_lines.append(
        f"{model_name} & "
        f"{accuracy} & "
        f"{precision} & "
        f"{recall} & "
        f"{f1} & "
        f"{roc_auc_value} \\\\"
    )

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_output = "\n".join(latex_lines)

latex_path = OUTPUT_DIR / "model_comparison_table.tex"

latex_path.write_text(
    latex_output,
    encoding="utf-8"
)

print(f"Saved: {latex_path}")

# Complete
print("\n"+"="*75)
print("PUBLICATION TABLE GENERATION COMPLETE")
print("="*75)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(f"  - {file.name}")

print("\nNo model was retrained.")
print("No official test data was modified.")
print("No threshold was tuned.")

print("="*75)