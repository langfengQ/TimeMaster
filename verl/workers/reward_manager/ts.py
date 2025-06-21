from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict

def per_class_accuracy(label_list, pred_list):
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for true_label, pred_label in zip(label_list, pred_list):
        total_per_class[true_label] += 1
        if true_label == pred_label:
            correct_per_class[true_label] += 1


    class_ids = sorted(set(label_list))  
    acc_per_class = []
    for class_id in class_ids:
        total = total_per_class[class_id]
        correct = correct_per_class[class_id]
        acc = correct / total if total > 0 else 0.0
        acc_per_class.append(acc)

    return acc_per_class
class TimeSeriesReward:
    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.buffer = np.array((100,), dtype=bool)
        self.save_idx = 0

    def verify(self, data):
      
        pass

    def print_format_success_rate(self):
        if self.save_idx >= 100:
            print("[Format success rate]: ", self.buffer.mean())

    def __call__(self, data: DataProto):


        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        pred_list = []
        label_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score, valid, pred, label = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            self.buffer[self.save_idx % len(self.buffer)] = valid
            self.save_idx += 1


            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
                print("index: ", data_item.non_tensor_batch['index'])
            pred_list.append(pred)
            label_list.append(label)
            

        #####

        f1_scores = f1_score(label_list, pred_list, average='macro')
        acc = accuracy_score(label_list, pred_list)


        return reward_tensor, f1_scores,acc