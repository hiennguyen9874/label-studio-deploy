# Phase 3: API tests for prompt modes and routing

**Goal:** Build deterministic API-level tests that validate end-to-end backend behavior without requiring heavy SAM3 inference in test runtime.

**Tasks:** 2 related tasks.

### Task 1: Build request fixtures and inference mocking strategy

**Files:**
- Modify: `label_studio_ml/sam3/test_api.py`
- Create: `label_studio_ml/sam3/tests/fixtures.py`
- Modify: `label_studio_ml/sam3/requirements-test.txt`

- [ ] **Step 1: Write failing fixture import tests**

```python
# label_studio_ml/sam3/test_api.py
from tests.fixtures import make_task_payload, make_context_payload


def test_fixture_generates_box_context():
    ctx = make_context_payload(mode="box")
    assert ctx["result"][0]["type"] == "rectanglelabels"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_fixture_generates_box_context`
Expected: FAIL because fixture module does not exist.

- [ ] **Step 3: Add fixtures and deterministic mocks**

```python
# label_studio_ml/sam3/tests/fixtures.py

def make_task_payload(image_url="http://localhost/image.jpg"):
    return {"tasks": [{"id": 1, "data": {"image": image_url}}]}


def make_context_payload(mode="text"):
    if mode == "box":
        return {"result": [{"type": "rectanglelabels", "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["defect"]}, "original_width": 1000, "original_height": 1000}]}
    if mode == "keypoint":
        return {"result": [{"type": "keypointlabels", "value": {"x": 10, "y": 10, "keypointlabels": ["defect"]}, "is_positive": 1, "original_width": 1000, "original_height": 1000}]}
    return {"result": [{"type": "textarea", "value": {"text": ["ear"]}, "original_width": 1000, "original_height": 1000}]}
```

- [ ] **Step 4: Add test dependencies**

```text
# label_studio_ml/sam3/requirements-test.txt
pytest
pytest-cov
```

- [ ] **Step 5: Run fixture tests**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_fixture_generates_box_context`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add label_studio_ml/sam3/tests/fixtures.py label_studio_ml/sam3/test_api.py label_studio_ml/sam3/requirements-test.txt
git commit -m "test(sam3): add reusable API payload fixtures"
```

### Task 2: Add end-to-end `/predict` API cases by prompt mode and priority

**Files:**
- Modify: `label_studio_ml/sam3/test_api.py`

- [ ] **Step 1: Write `/predict` integration-style tests with monkeypatching**

```python
# label_studio_ml/sam3/test_api.py
import json
import numpy as np


def test_predict_box_mode(client, monkeypatch):
    request_payload = {
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "context": {"result": [{"type": "rectanglelabels", "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["defect"]}, "original_width": 1000, "original_height": 1000}]},
    }
    response = client.post('/predict', data=json.dumps(request_payload), content_type='application/json')
    assert response.status_code == 200
    assert json.loads(response.data)["results"][0]["result"][0]["type"] == "brushlabels"


def test_predict_keypoint_mode(client, monkeypatch):
    request_payload = {
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "context": {"result": [{"type": "keypointlabels", "value": {"x": 10, "y": 10, "keypointlabels": ["defect"]}, "is_positive": 1, "original_width": 1000, "original_height": 1000}]},
    }
    response = client.post('/predict', data=json.dumps(request_payload), content_type='application/json')
    assert response.status_code == 200


def test_predict_text_mode(client, monkeypatch):
    request_payload = {
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "context": {"result": [{"type": "textarea", "value": {"text": ["ear"]}, "original_width": 1000, "original_height": 1000}]},
    }
    response = client.post('/predict', data=json.dumps(request_payload), content_type='application/json')
    assert response.status_code == 200


def test_predict_mixed_prompts_uses_box_priority(client, monkeypatch):
    request_payload = {
        "tasks": [{"id": 1, "data": {"image": "http://localhost/image.jpg"}}],
        "context": {
            "result": [
                {"type": "textarea", "value": {"text": ["ear"]}, "original_width": 1000, "original_height": 1000},
                {"type": "keypointlabels", "value": {"x": 10, "y": 10, "keypointlabels": ["defect"]}, "is_positive": 1, "original_width": 1000, "original_height": 1000},
                {"type": "rectanglelabels", "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rectanglelabels": ["defect"]}, "original_width": 1000, "original_height": 1000},
            ]
        },
    }
    response = client.post('/predict', data=json.dumps(request_payload), content_type='application/json')
    assert response.status_code == 200
```

- [ ] **Step 2: Run tests to verify at least one fails first**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py -k "predict_"`
Expected: FAIL until model methods are patched and response assertions match the backend output schema.

- [ ] **Step 3: Complete assertions for response schema and priority mode behavior**

```python
assert "results" in payload
pred = payload["results"][0]
assert pred["model_version"].startswith("NewModel")
assert pred["result"][0]["value"]["format"] == "rle"
```

- [ ] **Step 4: Run full SAM3 backend tests**

Run: `cd label_studio_ml/sam3 && pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add label_studio_ml/sam3/test_api.py
git commit -m "test(sam3): validate predict modes and priority routing"
```

## Acceptance Criteria
- Tests cover text, keypoint, and box modes plus mixed-priority behavior.
- Tests do not require downloading large model checkpoints.
- `/predict` response schema is asserted for valid brush/RLE outputs.

## Phase Verification
Run: `cd label_studio_ml/sam3 && pytest -q`
Expected: PASS
