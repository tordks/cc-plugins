```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
```


```python
using_colab = False
```


```python
if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    !{sys.executable} -m pip install opencv-python matplotlib scikit-learn
    !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam3.git'
```


```python
import os
from glob import glob

import numpy as np
import sam3.visualization_utils as utils

from matplotlib import pyplot as plt

COLORS = utils.pascal_color_map()[1:]
```

1. Load the data


```python
# Preapre the data path
ANNOT_DIR = None # PUT YOUR ANNOTATION PATH HERE
IMG_DIR = None # PUT YOUR IMAGE PATH HERE

# Load the SA-CO/Gold annotation files
annot_file_list = glob(os.path.join(ANNOT_DIR, "*gold*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)
```

Show the annotation files being loaded


```python
annot_dfs.keys()
```

2. Examples of the data format


```python
annot_dfs["gold_fg_sports_equipment_merged_a_release_test"].keys()
```


```python
annot_dfs["gold_fg_sports_equipment_merged_a_release_test"]["info"]
```


```python
annot_dfs["gold_fg_sports_equipment_merged_a_release_test"]["images"].head(3)
```


```python
annot_dfs["gold_fg_sports_equipment_merged_a_release_test"]["annotations"].head(3)
```

3. Visualize the data


```python
# Select a target dataset
target_dataset_name = "gold_fg_food_merged_a_release_test"

import cv2
from pycocotools import mask as mask_util
from collections import defaultdict

# Group GT annotations by image_id
gt_image_np_pairs = annot_dfs[target_dataset_name]["images"]
gt_annotations = annot_dfs[target_dataset_name]["annotations"]

gt_image_np_map = {img["id"]: img for _, img in gt_image_np_pairs.iterrows()}
gt_image_np_ann_map = defaultdict(list)
for _, ann in gt_annotations.iterrows():
    image_id = ann["image_id"]
    if image_id not in gt_image_np_ann_map:
        gt_image_np_ann_map[image_id] = []
    gt_image_np_ann_map[image_id].append(ann)

positiveNPs = common_image_ids = [img_id for img_id in gt_image_np_map.keys() if img_id in gt_image_np_ann_map and gt_image_np_ann_map[img_id]]
negativeNPs = [img_id for img_id in gt_image_np_map.keys() if img_id not in gt_image_np_ann_map or not gt_image_np_ann_map[img_id]]

num_image_nps_to_show = 10
fig, axes = plt.subplots(num_image_nps_to_show, 3, figsize=(15, 5 * num_image_nps_to_show))
for idx in range(num_image_nps_to_show):
    rand_idx = np.random.randint(len(positiveNPs))
    image_id = positiveNPs[rand_idx]
    noun_phrase = gt_image_np_map[image_id]["text_input"]
    img_rel_path = gt_image_np_map[image_id]["file_name"]
    full_path = os.path.join(IMG_DIR, f"{img_rel_path}")
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_annotation = gt_image_np_ann_map[image_id]

    def display_image_in_subplot(img, axes, row, col, title=""):
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')


    noun_phrases = [noun_phrase]
    annot_masks = [mask_util.decode(ann["segmentation"]) for ann in gt_annotation]

    # Show the image
    display_image_in_subplot(img, axes, idx, 0, f"{noun_phrase}")

    # Show all masks over a white background
    all_masks = utils.draw_masks_to_frame(
        frame=np.ones_like(img)*255, masks=annot_masks, colors=COLORS[: len(annot_masks)]
    )
    display_image_in_subplot(all_masks, axes, idx, 1, f"{noun_phrase} - Masks only")

    # Show masks overlaid on the image
    masked_frame = utils.draw_masks_to_frame(
        frame=img, masks=annot_masks, colors=COLORS[: len(annot_masks)]
    )
    display_image_in_subplot(masked_frame, axes, idx, 2, f"{noun_phrase} - Masks overlaid")

```


```python

```
