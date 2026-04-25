"""
Step 5: Tạo co-occurrence matrix và paste compatibility mapping.
Tính P(j|i) từ train masks, dùng để kiểm soát copy-paste hợp ngữ cảnh.
"""
import ast
import numpy as np
import pandas as pd
from pathlib import Path

BACKGROUND_ID = 103
NUM_CLASSES = 104
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "processed"


def build_cooccurrence_graph():
    (OUT / "graph").mkdir(parents=True, exist_ok=True)

    sample_map = pd.read_csv(OUT / "metadata" / "sample_training_mapping.csv")
    class_map = pd.read_csv(OUT / "mappings" / "class_group_mapping.csv")
    id_to_name = dict(zip(class_map["class_id"], class_map["class_name"]))

    train_rows = sample_map[sample_map["split"] == "train"]

    cooc = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    presence = np.zeros(NUM_CLASSES, dtype=np.float32)

    print(f"[Step 5] Computing co-occurrence from {len(train_rows)} train samples...")
    for _, row in train_rows.iterrows():
        cls_list = ast.literal_eval(str(row["present_classes"]))
        cls = [int(c) for c in cls_list if int(c) != BACKGROUND_ID]
        for i in cls:
            presence[i] += 1
        for i in cls:
            for j in cls:
                if i != j:
                    cooc[i, j] += 1

    # P(j|i) — conditional probability
    cooc_prob = cooc / np.maximum(presence[:, None], 1.0)
    np.save(OUT / "graph" / "cooccurrence_matrix.npy", cooc_prob)
    print(f"  Saved cooccurrence_matrix.npy")

    # Build compatibility CSV
    rows = []
    for i in range(NUM_CLASSES):
        if i == BACKGROUND_ID:
            continue
        for j in range(NUM_CLASSES):
            if j == BACKGROUND_ID or i == j:
                continue
            score = float(cooc_prob[i, j])
            if score <= 0:
                level = "none"
            elif score < 0.05:
                level = "medium"
            else:
                level = "high"

            rows.append({
                "source_class_id": i,
                "source_class_name": id_to_name.get(i, f"class_{i}"),
                "target_context_class_id": j,
                "target_context_class_name": id_to_name.get(j, f"class_{j}"),
                "compatibility_score": round(score, 5),
                "compatibility_level": level,
            })

    df = pd.DataFrame(rows)
    out_path = OUT / "graph" / "paste_compatibility.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path} ({len(df)} pairs)")
    return df


if __name__ == "__main__":
    build_cooccurrence_graph()
