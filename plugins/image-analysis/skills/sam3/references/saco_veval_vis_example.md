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
import utils

from matplotlib import pyplot as plt

COLORS = utils.pascal_color_map()[1:]
```

1. Load the data


```python
# Preapre the data path
DATA_DIR = "./sam3_saco_veval_data" # PUT YOUR DATA PATH HERE
ANNOT_DIR = os.path.join(DATA_DIR, "annotation")

# Load the SACO/Veval annotation files
annot_file_list = glob(os.path.join(ANNOT_DIR, "*veval*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)
```

Show the annotation files being loaded


```python
annot_dfs.keys()
```

2. Examples of the data format


```python
annot_dfs["saco_veval_yt1b_val"].keys()
```


```python
annot_dfs["saco_veval_yt1b_val"]["info"]
```


```python
annot_dfs["saco_veval_yt1b_val"]["videos"].head(3)
```


```python
annot_dfs["saco_veval_yt1b_val"]["annotations"].head(3)
```


```python
annot_dfs["saco_veval_yt1b_val"]["categories"].head(3)
```


```python
annot_dfs["saco_veval_yt1b_val"]["video_np_pairs"].head(3)
```

3. Visualize the data


```python
# Select a target dataset
target_dataset_name = "saco_veval_yt1b_val"

# visualize a random positive video-np pair
df_pairs = annot_dfs[target_dataset_name]["video_np_pairs"]
df_positive_pairs = df_pairs[df_pairs.num_masklets > 0]
rand_idx = np.random.randint(len(df_positive_pairs))
pair_row = df_positive_pairs.iloc[rand_idx]
video_id = pair_row.video_id
noun_phrase = pair_row.noun_phrase
print(f"Randomly selected video-np pair: video_id={video_id}, noun_phrase={noun_phrase}")

def display_image_in_subplot(img, axes, row, col, title=""):
    axes[row, col].imshow(img)
    axes[row, col].set_title(title)
    axes[row, col].axis('off')

num_frames_to_show = 5  # Number of frames to show per dataset
every_n_frames = 4  # Interval between frames to show

fig, axes = plt.subplots(num_frames_to_show, 3, figsize=(15, 5 * num_frames_to_show))

for idx in range(0, num_frames_to_show):
    sampled_frame_idx = idx * every_n_frames
    print(f"Reading annotations for frame {sampled_frame_idx}")
    # Get the frame and the corresponding masks and noun phrases
    frame, annot_masks, annot_noun_phrases = utils.get_all_annotations_for_frame(
        annot_dfs[target_dataset_name], video_id=video_id, frame_idx=sampled_frame_idx, data_dir=DATA_DIR, dataset=target_dataset_name
    )
    # Filter masks and noun phrases by the selected noun phrase
    annot_masks = [m for m, np in zip(annot_masks, annot_noun_phrases) if np == noun_phrase]

    # Show the frame
    display_image_in_subplot(frame, axes, idx, 0, f"{target_dataset_name} - {noun_phrase} - Frame {sampled_frame_idx}")

    # Show the annotated masks
    if annot_masks is None:
        print(f"No masks found for video_id {video_id} at frame {sampled_frame_idx}")
    else:
        # Show all masks over a white background
        all_masks = utils.draw_masks_to_frame(
            frame=np.ones_like(frame)*255, masks=annot_masks, colors=COLORS[: len(annot_masks)]
        )
        display_image_in_subplot(all_masks, axes, idx, 1, f"{target_dataset_name} - {noun_phrase} - Frame {sampled_frame_idx} - Masks")
        
        # Show masks overlaid on the frame
        masked_frame = utils.draw_masks_to_frame(
            frame=frame, masks=annot_masks, colors=COLORS[: len(annot_masks)]
        )
        display_image_in_subplot(masked_frame, axes, idx, 2, f"Dataset: {target_dataset_name} - {noun_phrase} - Frame {sampled_frame_idx} - Masks overlaid")

plt.tight_layout()
plt.show()
```


```python

```
