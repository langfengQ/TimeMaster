# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        # from . import math
        # res = math.compute_score(solution_str, ground_truth)

        # Use Math-Verify (https://github.com/huggingface/Math-Verify) for better evaluation accuracy
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['finance_SP_dataset']:
        from . import finance
        res, valid, pred, true = finance.compute_score(solution_str, ground_truth)
    elif data_source in ['har_dataset']:
        from . import har
        res, valid, pred, true = har.compute_score(solution_str, ground_truth)
    elif data_source in ['lightning_dataset']:
        from . import lightning
        res, valid, pred, true = lightning.compute_score(solution_str, ground_truth)
    elif data_source in ['emg_dataset']:
        from . import emg
        res, valid, pred, true = emg.compute_score(solution_str, ground_truth)
    elif data_source in ['computer_dataset']:
        from . import ctu
        res, valid, pred, true = ctu.compute_score(solution_str, ground_truth)
    elif data_source in ['ecg_dataset']:
        from . import ecg
        res, valid, pred, true = ecg.compute_score(solution_str, ground_truth)
    elif data_source in ['rcw_dataset']:
        from . import rcw
        res, valid, pred, true = rcw.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
    try:
        if isinstance(res, (int, float, bool)):
            return float(res), int(valid),pred, true
        else:
            return float(res[0]),int(valid),pred, true
    except:
        if isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
