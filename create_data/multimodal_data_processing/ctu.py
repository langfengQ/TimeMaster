import os
import argparse
import random
from collections import defaultdict

from datasets import load_from_disk, Dataset
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs


def make_map_fn(split):
    instruction_following = (
        "You MUST first perform step-by-step reasoning for your prediction. "
        "This reasoning MUST be enclosed within <think> </think> tags.\n"
        "After completing your reasoning, you must select ONE most appropriate label and enclose it within <class> </class> tags:\n"
        "\"Desktop\" or \"Laptop\".\n"
    )

    def process_fn(example, idx):
        # 拼接 final prompt
        problem = example.pop('problem')
        prompt = problem + "\n" + instruction_following
        answer = example.pop('answer')
        if answer not in ['Desktop', 'Laptop']:
            raise ValueError(f"Invalid answer: {answer}")
        image_path = example.pop('image')

        if 'train' in image_path:
            image_path = './data/ctu_image/images/train/' + image_path
        else:
            image_path = './data/ctu_image/images/test/' + image_path

  
        image = Image.open(image_path).convert("RGBA")

        data = {
            "data_source": "computer_dataset",
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "images": [image],
            "ability": "time series",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer,
                "question": problem,
            }
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/ctu_image')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir, exist_ok=True)

   
    dataset = load_from_disk(os.path.expanduser(args.local_dir))

  
    train_raw = list(dataset['train'])
    test_raw = list(dataset['test'])

    train_dataset = Dataset.from_list(train_raw)
    test_dataset = Dataset.from_list(test_raw)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)


    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))


    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
