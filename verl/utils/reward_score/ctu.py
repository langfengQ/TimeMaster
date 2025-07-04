import re
import json


label_map = {
    0: "desktop",
    1: "laptop",

}

# Reverse map: label name (lowercased) to id
label_dict = {v.lower(): k for k, v in label_map.items()}


def extract_prediction(text: str) -> str:

    match = re.search(
        r'<class>\s*(desktop|laptop)\s*</class>',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip().lower()
    return "None"




def acc_reward(predict_str: str, ground_truth: str) -> tuple[float, float, int, int]:
    answer = extract_prediction(predict_str)
    ground_truth = ground_truth.strip().lower()

    if ground_truth not in label_dict:
        print(f"❌ Invalid ground truth: {ground_truth}")
        return 0.0, 0.0, -1, -1

    label_id = label_dict[ground_truth]

    if answer == "None":
        pred_id = next(v for v in label_dict.values() if v != label_id)  # fake incorrect pred
        return 0.0, 0.0, pred_id, label_id

    pred_id = label_dict.get(answer, -1)
    correct = 1.0 if pred_id == label_id else 0.0

    return correct, 1.0, pred_id, label_id


def compute_score(predict_str: str, ground_truth: str) -> tuple[float, float, int, int]:
    acc, fmt, pred_id, label_id = acc_reward(predict_str, ground_truth)

    return 0.9 * acc + 0.1 * fmt, fmt, pred_id, label_id
