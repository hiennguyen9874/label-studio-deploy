# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %%
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# %%
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.4)

# %%
import requests

image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image.size # (640, 425)

# %% [markdown]
# ## Text-Only Prompts

# %%
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
import matplotlib.pyplot as plt
import numpy as np

with torch.autocast("cuda"):
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="ear")

print(inference_state.keys()) # dict_keys(['original_height', 'original_width', 'backbone_out', 'geometric_prompt', 'masks_logits', 'masks', 'boxes', 'scores'])
print(inference_state['boxes'].shape) # [2, 4]
print(inference_state['masks_logits'].shape) # [2, 1, 425, 640]
print(inference_state['masks'].shape) # [2, 1, 425, 640]
print(inference_state['scores']) # [0.9048, 0.9180]

img0 = image.copy()
plot_results(img0, inference_state)
plt.axis('off')
plt.show()

# %% [markdown]
# ## Single Bounding Box Prompt

# %%
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
import matplotlib.pyplot as plt
import numpy as np

from sam3.model.box_ops import box_xywh_to_cxcywh

width, height = image.size

# Here the box is in  (x,y,w,h) format, where (x,y) is the top left corner.
box_input_xywh = torch.tensor([100, 150, 500, 450]).view(-1, 4)
box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
print("Normalized box input:", norm_box_cxcywh)

with torch.autocast("cuda"):
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.add_geometric_prompt(
        state=inference_state, box=norm_box_cxcywh, label=True
    )

print(inference_state.keys()) # dict_keys(['original_height', 'original_width', 'backbone_out', 'geometric_prompt', 'masks_logits', 'masks', 'boxes', 'scores'])
print(inference_state['boxes'].shape) # [2, 4]
print(inference_state['masks_logits'].shape) # [2, 1, 425, 640]
print(inference_state['masks'].shape) # [2, 1, 425, 640]
print(inference_state['scores']) # [0.9048, 0.9180]

img0 = image.copy()
plot_results(img0, inference_state)
plt.axis('off')
plt.show()

# %% [markdown]
# ## Multiple Box Prompts (Positive and Negative)

# %%
# Segment "handle" but exclude the oven handle using a negative box
text = "handle"
# Negative box covering oven handle area (xyxy): [40, 183, 318, 204]
oven_handle_box = [40, 183, 318, 204]
input_boxes = [[oven_handle_box]]


processor.reset_all_prompts(inference_state)
with torch.autocast("cuda"):
    inference_state = processor.set_image(image)
    for box, label in zip(norm_boxes_cxcywh, box_labels):
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=box, label=True
        )

print(inference_state.keys()) # dict_keys(['original_height', 'original_width', 'backbone_out', 'geometric_prompt', 'masks_logits', 'masks', 'boxes', 'scores'])
print(inference_state['boxes'].shape) # [2, 4]
print(inference_state['masks_logits'].shape) # [2, 1, 425, 640]
print(inference_state['masks'].shape) # [2, 1, 425, 640]
print(inference_state['scores']) # [0.9048, 0.9180]

img0 = image.copy()
plot_results(img0, inference_state)
plt.axis('off')
plt.show()

# %%
