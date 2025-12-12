```python
import json
import os

from sam3.eval.saco_veval_eval import VEvalEvaluator
```


```python
DATASETS_TO_EVAL = [
    "saco_veval_sav_test",
    "saco_veval_yt1b_test",
    "saco_veval_smartglasses_test",
]
# Update to the directory where the GT annotation and PRED files exist
GT_DIR = None # PUT YOUR ANNOTATION PATH HERE
PRED_DIR = None # PUT YOUR IMAGE PATH HERE
```


```python
all_eval_res = {}
for dataset_name in DATASETS_TO_EVAL:
    gt_annot_file = os.path.join(GT_DIR, dataset_name + ".json")
    pred_file = os.path.join(PRED_DIR, dataset_name + "_preds.json")
    eval_res_file = os.path.join(PRED_DIR, dataset_name + "_eval_res.json")

    if os.path.exists(eval_res_file):
        with open(eval_res_file, "r") as f:
            eval_res = json.load(f)
    else:
        # Alternatively, we can run the evaluator offline first
        # by leveraging sam3/eval/saco_veval_eval.py
        print(f"=== Running evaluation for Pred {pred_file} vs GT {gt_annot_file} ===")
        veval_evaluator = VEvalEvaluator(
            gt_annot_file=gt_annot_file, eval_res_file=eval_res_file
        )
        eval_res = veval_evaluator.run_eval(pred_file=pred_file)
        print(f"=== Results saved to {eval_res_file} ===")

    all_eval_res[dataset_name] = eval_res
```


```python
REPORT_METRICS = {
    "video_mask_demo_cgf1_micro_50_95": "cgf1",
    "video_mask_all_phrase_HOTA": "pHOTA",
}
```


```python
res_to_print = []
for dataset_name in DATASETS_TO_EVAL:
    eval_res = all_eval_res[dataset_name]
    row = [dataset_name]
    for metric_k, metric_v in REPORT_METRICS.items():
        row.append(eval_res["dataset_results"][metric_k])
    res_to_print.append(row)

# Print dataset header (each dataset spans 2 metrics: 13 + 3 + 13 = 29 chars)
print("| " + " | ".join(f"{ds:^29}" for ds in DATASETS_TO_EVAL) + " |")

# Print metric header
metrics = list(REPORT_METRICS.values())
print("| " + " | ".join(f"{m:^13}" for _ in DATASETS_TO_EVAL for m in metrics) + " |")

# Print eval results
values = []
for row in res_to_print:
    values.extend([f"{v * 100:^13.1f}" for v in row[1:]])
print("| " + " | ".join(values) + " |")
```


```python

```
