# Phase 2: Prompt parsing, priority routing, and SAM3 inference

**Goal:** Implement interactive prediction with `TextArea`, `KeyPointLabels`, and `RectangleLabels` using strict prompt priority and brush mask outputs.

**Tasks:** 3 related tasks.

### Task 1: Parse context payload and normalize prompt inputs

**Files:**
- Modify: `label_studio_ml/sam3/model.py`
- Reference: `label_studio_ml/examples/segment_anything_2_image/model.py`
- Reference: `label_studio_ml/examples/grounding_sam/dino.py`

- [ ] **Step 1: Write failing parsing tests (text, keypoint, rectangle)**

```python
# label_studio_ml/sam3/test_api.py
from model import NewModel


def test_parse_textarea_prompt_payload():
    model = NewModel()
    context = {"result": [{"type": "textarea", "value": {"text": ["ear"]}}]}
    parsed = model._parse_prompts(context=context, image_width=1000, image_height=1000)
    assert parsed["text"] == "ear"


def test_parse_keypoints_payload():
    model = NewModel()
    context = {
        "result": [{
            "type": "keypointlabels",
            "value": {"x": 10, "y": 12, "keypointlabels": ["defect"]},
            "is_positive": 1,
        }]
    }
    parsed = model._parse_prompts(context=context, image_width=1000, image_height=1000)
    assert parsed["points"] == [[100, 120]]
    assert parsed["point_labels"] == [1]


def test_parse_rectangle_payload():
    model = NewModel()
    context = {
        "result": [{
            "type": "rectanglelabels",
            "value": {"x": 10, "y": 12, "width": 20, "height": 16, "rectanglelabels": ["defect"]},
        }]
    }
    parsed = model._parse_prompts(context=context, image_width=1000, image_height=1000)
    assert parsed["box_xyxy"] == [100, 120, 300, 280]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "parse_"`
Expected: FAIL because parsing helpers are not implemented.

- [ ] **Step 3: Implement context parsing helpers in model**

```python
# in NewModel

def _parse_prompts(self, context, image_width, image_height):
    text_prompt = None
    point_coords = []
    point_labels = []
    input_box = None

    for ctx in context.get("result", []):
        ctx_type = ctx.get("type")
        value = ctx.get("value", {})

        if "text" in value and value["text"]:
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
```

- [ ] **Step 4: Run tests to verify parser behavior**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "parse_"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add label_studio_ml/sam3/model.py label_studio_ml/sam3/test_api.py
git commit -m "feat(sam3): add prompt parsing and coordinate normalization"
```

### Task 2: Implement priority routing and per-mode SAM3 inference

**Files:**
- Modify: `label_studio_ml/sam3/model.py`

- [ ] **Step 1: Write failing routing tests**

```python
# label_studio_ml/sam3/test_api.py
from model import NewModel


def test_priority_uses_box_over_keypoint_and_text():
    model = NewModel()
    parsed = {
        "text": "ear",
        "points": [[10, 20]],
        "point_labels": [1],
        "box_xyxy": [10, 20, 30, 40],
    }
    assert model._select_prompt_mode(parsed) == "box"


def test_priority_uses_keypoint_over_text_when_no_box():
    model = NewModel()
    parsed = {
        "text": "ear",
        "points": [[10, 20]],
        "point_labels": [1],
        "box_xyxy": None,
    }
    assert model._select_prompt_mode(parsed) == "keypoint"


def test_priority_uses_text_when_only_text_present():
    model = NewModel()
    parsed = {
        "text": "ear",
        "points": [],
        "point_labels": [],
        "box_xyxy": None,
    }
    assert model._select_prompt_mode(parsed) == "text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "priority_"`
Expected: FAIL because selector/inference dispatcher is not implemented.

- [ ] **Step 3: Implement selector + dispatch logic**

```python
# in NewModel

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
        return None

    # use SAM3 processor API as defined by installed `sam3` package
    state = self._processor.set_image(image)
    self._processor.reset_all_prompts(state)

    if mode == "box":
        state = self._processor.add_geometric_prompt(state=state, box=parsed["box_xyxy"], label=True)
    elif mode == "keypoint":
        for point, label in zip(parsed["points"], parsed["point_labels"]):
            state = self._processor.add_geometric_prompt(state=state, point=point, label=bool(label))
    else:
        state = self._processor.set_text_prompt(state=state, prompt=parsed["text"])

    return state
```

- [ ] **Step 4: Run tests to verify routing behavior**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "priority_"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add label_studio_ml/sam3/model.py label_studio_ml/sam3/test_api.py
git commit -m "feat(sam3): enforce box-keypoint-text priority routing"
```

### Task 3: Format predictions as BrushLabels RLE and finalize predict flow

**Files:**
- Modify: `label_studio_ml/sam3/model.py`

- [ ] **Step 1: Write failing output-shape tests**

```python
# label_studio_ml/sam3/test_api.py
import json


def test_predict_returns_brushlabels_rle(client, monkeypatch):
    request = {
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "label_config": "<View><Image name='image' value='$image'/><BrushLabels name='brush' toName='image'><Label value='defect'/></BrushLabels><RectangleLabels name='rect' toName='image' smart='true'><Label value='defect'/></RectangleLabels></View>",
        "context": {"result": [{"type": "rectanglelabels", "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["defect"]}, "original_width": 1000, "original_height": 1000}]}
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    payload = json.loads(response.data)
    result = payload["results"][0]["result"][0]
    assert result["type"] == "brushlabels"
    assert "rle" in result["value"]


def test_predict_returns_empty_when_context_missing(client):
    request = {
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "label_config": "<View><Image name='image' value='$image'/><BrushLabels name='brush' toName='image'><Label value='defect'/></BrushLabels></View>"
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    payload = json.loads(response.data)
    assert payload["results"][0]["result"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "brushlabels|context_missing"`
Expected: FAIL until output formatter and predict flow are complete.

- [ ] **Step 3: Implement output conversion and `predict()` orchestration**

```python
# in NewModel
from label_studio_sdk.converter import brush
from uuid import uuid4


def _to_brush_prediction(self, mask, score, width, height, from_name, to_name, label):
    rle = brush.mask2rle((mask.astype("uint8") * 255))
    return {
        "result": [{
            "id": str(uuid4())[:9],
            "from_name": from_name,
            "to_name": to_name,
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "type": "brushlabels",
            "value": {"format": "rle", "rle": rle, "brushlabels": [label]},
            "score": float(score),
            "readonly": False,
        }],
        "score": float(score),
        "model_version": self.get("model_version"),
    }
```

- [ ] **Step 4: Run focused predict tests**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "predict|brushlabels|context_missing"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add label_studio_ml/sam3/model.py label_studio_ml/sam3/test_api.py
git commit -m "feat(sam3): return SAM3 masks as BrushLabels predictions"
```

## Acceptance Criteria
- `predict()` supports text-only, keypoint-only, and box-only interactions.
- Mixed inputs apply strict priority: box > keypoint > text.
- Responses are valid `BrushLabels` RLE payloads with model version and score.
- Missing context safely returns empty predictions.

## Phase Verification
Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "parse_|priority_|predict|brushlabels|context_missing"`
Expected: PASS
