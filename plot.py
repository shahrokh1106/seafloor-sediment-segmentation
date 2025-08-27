from glob import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    results_path = os.path.join("results", "phase-3")
    
    results_path_dict = {
        "0": os.path.join(results_path, "SegformerDatasetV00.pth","SegformerDatasetV00_metrics.json"),
        "1": os.path.join(results_path, "SegformerDatasetV1.pth","SegformerDatasetV1_metrics.json"),
        "2": os.path.join(results_path, "SegformerDatasetV2.pth","SegformerDatasetV2_metrics.json"),
        "3": os.path.join(results_path, "SegformerDatasetV3.pth","SegformerDatasetV3_metrics.json"),
    }

    results_dict = {}
    for iter,path in results_path_dict.items():
        with open(path, "r") as f:
            data = json.load(f)
        results_dict.update({iter:data})
    iters = sorted(results_dict.keys(), key=lambda x: int(x))
    
    label_map = results_dict[iters[0]]["label_map"]  # list of class names
    classes = label_map  # keep background; drop later if you prefer

   
    df_iou = pd.DataFrame(
        {it: results_dict[it]["per_category_iou"] for it in iters},
        index=classes
    )
    df_pa = pd.DataFrame(
        {it: results_dict[it]["pixel_accuracy_per_class"] for it in iters},
        index=classes
    )

    
    df_summary = pd.DataFrame({
        "mIoU": [results_dict[it]["mean_iou"] for it in iters],
        "mPA":  [results_dict[it].get("mean_pixel_accuracy", results_dict[it]["mean_class_accuracy"]) for it in iters],
        "mIoU_weighted": [results_dict[it].get("mean_iou_weighted", np.nan) for it in iters],
        "PA_weighted":   [results_dict[it].get("pixel_accuracy_weighted", np.nan) for it in iters],
    }, index=iters)

    
    # ---------- Plot 5: Grouped bars (mean vs weighted) ----------
    bar_df = df_summary[["mIoU", "mIoU_weighted", "mPA", "PA_weighted"]].copy()
    plt.figure()
    x = np.arange(len(bar_df.index))
    w = 0.2
    plt.bar(x - 1.5*w, bar_df["mIoU"], width=w, label="mIoU")
    plt.bar(x - 0.5*w, bar_df["mIoU_weighted"], width=w, label="mIoU (weighted)")
    plt.bar(x + 0.5*w, bar_df["mPA"], width=w, label="mPA")
    plt.bar(x + 1.5*w, bar_df["PA_weighted"], width=w, label="mPA (weighted)")
    plt.xticks(x, bar_df.index)
    plt.ylabel("Score")
    plt.title("Mean vs Weighted Metrics by Iteration")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


    

