# Copyright 2025 Zhejiang University (ZJU), China and TimeMaster Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import time
from openai import OpenAI

# EMG label mapping
label_map = {
    0: "Healthy",
    1: "Myopathy",
    2: "Neuropathy"
}
label_dict = {v.lower(): k for k, v in label_map.items()}

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def extract_tag_content(text: str, tag: str) -> str:
    """Extract content inside a custom XML-like tag."""
    pattern = fr"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def acc_reward_emg(predict_str: str, ground_truth: str) -> tuple[float, float, int, int]:
    """
    Check if the predicted class (inside <class>) matches the ground truth.
    """
    prediction = extract_tag_content(predict_str, "class").lower()
    ground_truth = ground_truth.strip().lower()

    if ground_truth not in label_dict:
        print(f"[acc_reward_emg] ❌ Invalid ground truth: {ground_truth}")
        return 0.0, 0.0, -1, -1

    label_id = label_dict[ground_truth]
    pred_id = label_dict.get(prediction, -1)

    if pred_id == -1:
        return 0.0, 0.0, -1, label_id

    acc = 1.0 if pred_id == label_id else 0.0
    fmt = 1.0 if prediction else 0.0
    return acc, fmt, pred_id, label_id

def penalize_if_generic(extension_text: str, raw_score: float) -> float:
    """
    Apply penalty if the extension is too generic.
    """
    generic_phrases = [
        r"further (diagnostic )?tests",
        r"clinical history",
        r"physical examination",
        r"refer to specialist",
        r"confirm the diagnosis"
    ]
    if any(re.search(pat, extension_text.lower()) for pat in generic_phrases):
        return min(raw_score, 0.4)
    return raw_score

def score_extension_with_qwen(extension_text: str, prediction: str, reasoning: str) -> float:
    """
    Ask the model to evaluate the quality of the extension.
    """
    prompt = (
        "You are a clinical evaluation assistant. The following is a model-generated diagnostic reasoning, prediction, "
        "and clinical recommendation (called 'extension').\n\n"
        f"Reasoning: {reasoning}\n"
        f"Prediction: {prediction}\n"
        f"Extension (clinical recommendation): {extension_text}\n\n"
        "Evaluate the clinical recommendation (extension) based on the following four dimensions. For each, assign a score between 0.0 and 1.0. "
        "After scoring, return ONLY the **average score** as a single float (e.g., 0.625).\n\n"
        "Scoring dimensions:\n"
        "1. Specificity – Is the recommendation tailored to the case, or is it generic advice?\n"
        "2. Appropriateness – Is it clinically sound and aligned with the predicted condition?\n"
        "3. Relevance – Does it logically follow from the reasoning and prediction?\n"
        "4. Depth – Does it show clinical judgment and insight?\n\n"
        "Return only the average score as a float between 0.0 and 1.0."
    )

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )
            result = response.choices[0].message.content.strip()
            score = float(result)
            score = penalize_if_generic(extension_text, score)
            if 0.0 <= score <= 1.0:
                return score
            else:
                print(f"[score_extension] ❌ Invalid score: {result}")
        except Exception as e:
            if "429" in str(e) or "RATE_LIMIT_EXCEEDED" in str(e):
                print("[Rate Limited] Waiting before retrying...")
                time.sleep(21)
            else:
                print(f"[score_extension] Failed: {e}")
                break
    return 0.0

def compute_score(predict_str: str, ground_truth: str) -> tuple[float, float, int, int, float]:
    """
    Computes final score:
    - base reward (accuracy + format)
    - adds extension_score if correct
    """
    acc, fmt, pred_id, label_id = acc_reward_emg(predict_str, ground_truth)
    reward = 0.9 * acc + 0.1 * fmt
    extension_score = 0.0

    if acc == 1.0 and fmt == 1.0:
        reasoning = extract_tag_content(predict_str, "think")
        prediction = extract_tag_content(predict_str, "class")
        extension = extract_tag_content(predict_str, "extension")

        if extension:
            try:
                extension_score = score_extension_with_qwen(extension, prediction, reasoning)
                reward += extension_score * 0.5
                print(f"[compute_score] ✅ Extension score: {extension_score}")
            except Exception as e:
                print(f"[compute_score] ❌ Failed extension eval: {e}")
        else:
            print("Fail:", extension)

    return reward, fmt, pred_id, label_id
