# Copyright 2025 Zhejiang University (ZJU), China.
# Copyright 2025 TimeMaster team.
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

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
from collections import Counter
import arff

def read_arff_file(file_path):
    with open(file_path, 'r') as f:
        data = arff.load(f)
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    return df

label_map = {
    1: "Desktop",
    2: "Laptop"
}

def plot_signal(signal, label_str, out_prefix):
    plt.figure(figsize=(6, 2.5))
    plt.plot(signal, linewidth=2)
    plt.title("Complete Signal", fontsize=9)
    plt.xlabel("Timestamp", fontsize=8)
    plt.ylabel("Value", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    image_path = out_prefix + label_str.replace(" ", "_") + '.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    return os.path.basename(image_path)

def build_split_dataset(X, y, output_img_dir, split_name):
    os.makedirs(output_img_dir, exist_ok=True)
    problem_template = (
        "<image> You are analyzing a time series signal derived from electricity usage patterns in UK households, "
        "recorded as part of the government-sponsored study 'Powering the Nation'. The signal represents energy consumption "
        "sampled every 2 minutes over a 24-hour period, resulting in a series length of 720. The plot shows a segment of the household's daily electricity usage pattern.\n\n"
        "Your task is to classify the household's device usage pattern into one of the following two classes:\n"
        "- Desktop: The energy consumption pattern suggests the use of a desktop computer.\n"
        "- Laptop: The energy consumption pattern suggests the use of a laptop computer.\n"
        "Please choose the label that best matches the full signal."
    )
    examples = []
    for i in range(len(X)):
        signal = X[i]
        label_id = y[i]
        out_prefix = os.path.join(output_img_dir, f'{split_name}_sample_{i:05d}_')
        label_str = label_map[label_id]
        image_path = plot_signal(signal, label_str, out_prefix)
        example = {
            'problem': problem_template,
            'answer': label_str,
            'data': signal.tolist(),
            'image': image_path
        }
        examples.append(example)
    return Dataset.from_list(examples)

def main(args):
    # create folders
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'images', 'test'), exist_ok=True)

    data_path = Path(args.l7_dir)
    output_dir = Path(args.output_dir)
    train_df = read_arff_file(data_path / 'Computers_TRAIN.arff')
    test_df = read_arff_file(data_path / 'Computers_TEST.arff')
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    Y_train = train_df.iloc[:, -1].values.astype(int)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    Y_test = test_df.iloc[:, -1].values.astype(int)
    train_ds = build_split_dataset(X_train, Y_train, output_dir / 'images/train', 'train')
    test_ds = build_split_dataset(X_test, Y_test, output_dir / 'images/test', 'test')
    dataset = DatasetDict({
        'train': train_ds,
        'test': test_ds
    })
    dataset.save_to_disk(str(output_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l7_dir', type=str, default='./ori_data/Computers')
    parser.add_argument('--output_dir', type=str, default='./data/ctu_image')
    args = parser.parse_args()
    main(args)
