# Phase 1: Backend scaffold and runtime wiring

**Goal:** Create a runnable Label Studio ML backend skeleton for SAM3 with correct entrypoint and model lifecycle.

**Tasks:** 2 related tasks.

### Task 1: Create backend class shell with lazy SAM3 initialization

**Files:**
- Create: `label_studio_ml/sam3/model.py`
- Reference: `label_studio_ml/examples/segment_anything_2_image/model.py`
- Reference: `label_studio_ml/examples/grounding_sam/dino.py`

- [ ] **Step 1: Write the failing test for backend app boot**

```python
# label_studio_ml/sam3/test_api.py

def test_health_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_health_endpoint`
Expected: FAIL with import or missing backend module/class errors.

- [ ] **Step 3: Write minimal backend implementation shell**

```python
# label_studio_ml/sam3/model.py
import logging
from typing import List, Dict, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

class NewModel(LabelStudioMLBase):
    def setup(self):
        self.set('model_version', f'{self.__class__.__name__}-v0.0.1')
        self._model = None
        self._processor = None

    def _ensure_model_loaded(self):
        if self._model is not None and self._processor is not None:
            return
        # load SAM3 model + processor here in later phase

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        return ModelResponse(predictions=[])
```

- [ ] **Step 4: Run test to verify module imports pass**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_health_endpoint`
Expected: still FAIL until `_wsgi.py` is added, but no syntax/import errors from `model.py`.

- [ ] **Step 5: Commit**

```bash
git add label_studio_ml/sam3/model.py label_studio_ml/sam3/test_api.py
git commit -m "feat(sam3): scaffold Label Studio backend model class"
```

### Task 2: Add WSGI entrypoint and align runtime launch

**Files:**
- Create: `label_studio_ml/sam3/_wsgi.py`
- Modify: `label_studio_ml/sam3/start.sh`
- Modify: `label_studio_ml/sam3/docker-compose.yml`
- Modify: `label_studio_ml/sam3/Dockerfile` (only if path/entrypoint mismatches found)

- [ ] **Step 1: Write failing API bootstrap fixture**

```python
# label_studio_ml/sam3/test_api.py
import pytest
from model import NewModel

@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NewModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_health_endpoint`
Expected: FAIL because `_wsgi.py` does not exist yet.

- [ ] **Step 3: Implement `_wsgi.py` based on example pattern**

```python
# label_studio_ml/sam3/_wsgi.py
from label_studio_ml.api import init_app
from model import NewModel

app = init_app(model_class=NewModel)
```

- [ ] **Step 4: Ensure runtime command points to `_wsgi:app`**

```bash
# label_studio_ml/sam3/start.sh
exec gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 --pythonpath '/app' _wsgi:app
```

- [ ] **Step 5: Run test to verify backend starts in test mode**

Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_health_endpoint`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add label_studio_ml/sam3/_wsgi.py label_studio_ml/sam3/start.sh label_studio_ml/sam3/docker-compose.yml label_studio_ml/sam3/Dockerfile label_studio_ml/sam3/test_api.py
git commit -m "chore(sam3): wire WSGI entrypoint and runtime bootstrap"
```

## Acceptance Criteria
- `label_studio_ml/sam3` contains a runnable `NewModel` backend class and `_wsgi.py` entrypoint.
- `start.sh` launches gunicorn with `_wsgi:app`.
- Health endpoint test passes locally.

## Phase Verification
Run: `cd label_studio_ml/sam3 && pytest -q test_api.py::test_health_endpoint`
Expected: PASS
