import pandas as pd

def train_to_excel(results):
    rows = []

    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            for fold_result in model_results:
                fold_idx = fold_result["fold"]
                metrics = fold_result["fold_metrics"]

                rows.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold_idx,
                    **metrics,
                    # "Training Time": fold_result["training_time"],
                    # "Peak Memory": fold_result["peak_memory"]
                })

    df = pd.DataFrame(rows)
    df.to_excel("training_results.xlsx", index=False)