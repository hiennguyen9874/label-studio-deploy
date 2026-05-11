import logging

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk.converter import brush

logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):
    def setup(self):
        model_version = f"{self.__class__.__name__}-v0.0.1"
        self.set("model_version", model_version)
        self._model = None
        self._processor = None

    def _ensure_model_loaded(self):
        if self._model is not None and self._processor is not None:
            return

        self._model = build_sam3_image_model(
            checkpoint_path="/models/sam3.pt",
            bpe_path="/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        )
        self._processor = Sam3Processor(self._model, confidence_threshold=0.4)

    def _parse_prompts(self, context, image_width, image_height):
        text_prompt = None
        point_coords = []
        point_labels = []
        input_box = None

        for ctx in (context or {}).get("result", []):
            ctx_type = ctx.get("type")
            value = ctx.get("value", {})

            if value.get("text"):
                text_prompt = value["text"][0].strip()

            if ctx_type == "keypointlabels":
                x = value["x"] * image_width / 100
                y = value["y"] * image_height / 100
                point_coords.append([int(x), int(y)])
                point_labels.append(int(ctx.get("is_positive", 1)))

            if ctx_type == "rectanglelabels":
                x = value["x"] * image_width / 100
                y = value["y"] * image_height / 100
                w = value["width"] * image_width / 100
                h = value["height"] * image_height / 100
                input_box = [int(x), int(y), int(x + w), int(y + h)]

        return {
            "text": text_prompt,
            "points": point_coords,
            "point_labels": point_labels,
            "box_xyxy": input_box,
        }

    def _select_prompt_mode(self, parsed):
        if parsed["box_xyxy"] is not None:
            return "box"
        if parsed["points"]:
            return "keypoint"
        if parsed["text"]:
            return "text"
        return None

    def _run_sam3(self, image, parsed):
        mode = self._select_prompt_mode(parsed)
        if mode is None:
            return None, 0.0

        with torch.autocast("cuda"):
            state = self._processor.set_image(image)
            self._processor.reset_all_prompts(state)

            if mode == "box":
                state = self._processor.add_geometric_prompt(
                    state=state, box=parsed["box_xyxy"], label=True
                )
            elif mode == "keypoint":
                for point, label in zip(parsed["points"], parsed["point_labels"]):
                    state = self._processor.add_geometric_prompt(
                        state=state, point=point, label=bool(label)
                    )
            else:
                state = self._processor.set_text_prompt(
                    state=state, prompt=parsed["text"]
                )

        masks = state.get("masks")
        scores = state.get("scores")

        masks = masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        scores = scores.cpu().numpy()
        return masks, scores

    def _to_brush_prediction(
        self, masks, scores, width, height, from_name, to_name, label
    ):
        results = []
        total_score = 0

        print(masks.shape, scores.shape)

        for mask, score in zip(masks, scores):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:9]

            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            score = float(score)

            print("----------------------")
            print(mask)
            print(score)
            print("----------------------")

            results.append(
                {
                    "id": label_id,
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [label],
                    },
                    "score": score,
                    "type": "brushlabels",
                    "readonly": False,
                }
            )
            total_score += score
        return {
            "result": results,
            "score": total_score / max(len(results), 1),
            "model_version": self.get("model_version"),
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        model_version = self.get("model_version") or getattr(
            self, "model_version", f"{self.__class__.__name__}-v0.0.1"
        )

        if not tasks:
            return [{"result": [], "score": 0.0, "model_version": model_version}]

        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")

        if not context or not context.get("result"):
            return [{"result": [], "score": 0.0, "model_version": model_version}]

        task = tasks[0]
        image_url = task["data"][value]

        image_path = (
            "/external-data/" + image_url.split("?d=")[1]
            if "?d=" in image_url
            else self.get_local_path(image_url, task_id=task.get("id"))
        )
        image = Image.open(image_path).convert("RGB")

        try:
            image_width = context["result"][0]["original_width"]
            image_height = context["result"][0]["original_height"]
        except:
            image_width, image_height = image.size

        parsed = self._parse_prompts(
            context=context, image_width=image_width, image_height=image_height
        )

        mode = self._select_prompt_mode(parsed)
        if mode is None:
            return [{"result": [], "score": 0.0, "model_version": model_version}]

        self._ensure_model_loaded()

        masks, scores = self._run_sam3(image, parsed)

        label = None
        for ctx in context.get("result", []):
            values = ctx.get("value", {})
            for key in ("rectanglelabels", "keypointlabels", "text"):
                if values.get(key):
                    label = values[key][0]
                    break

        prediction = self._to_brush_prediction(
            masks, scores, image_width, image_height, from_name, to_name, label
        )
        return [prediction]
