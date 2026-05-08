# SAM3 ML Backend Design for Label Studio

## Request Summary
Build a Label Studio ML backend for SAM3 with support for three interactive prompt inputs:
- `TextArea` (text prompt)
- `KeyPointLabels` (point prompts)
- `RectangleLabels` (box prompts)

Output should be segmentation masks in Label Studio `BrushLabels` format.

## Context Reviewed
- `label_studio_ml/sam3/sam3.py` (currently demo/notebook-style inference script)
- `label_studio_ml/examples/segment_anything_2_image/model.py` (interactive point+box → brush pattern)
- `label_studio_ml/examples/grounding_sam/dino.py` (text prompt parsing pattern)
- `.pi/skills/label_studio_ml/SKILL.md` (project conventions and backend best practices)

## Decisions Captured
1. **Prompt conflict resolution:** Priority strategy
2. **Text input source:** `TextArea`

## Feature Size and Risk
- **Feature size:** **Medium** (max 5 implementation phases)
- **Risk classification:** **Architecture/API-sensitive**
  - Reason: introduces new backend behavior and prompt routing semantics in ML prediction path.

---

## Approaches Considered

### Approach 1 — Direct SAM3 backend class (Recommended)
Create `label_studio_ml/sam3/model.py` implementing `LabelStudioMLBase`, using existing examples as structural reference and SAM3 APIs for inference.

**Pros**
- Smallest complexity increase
- Aligns with existing backend architecture
- Easiest to test and maintain

**Cons**
- Requires careful context parsing + prompt-priority handling

### Approach 2 — Adapter around current `sam3.py`
Wrap existing `sam3.py` logic with a backend class and reuse code as-is where possible.

**Pros**
- Reuses current demo code

**Cons**
- `sam3.py` is notebook-style and not backend-structured
- Higher cleanup/normalization effort

### Approach 3 — Two-stage pipeline (text localization then segmentation)
Use text to derive region candidates first, then run SAM3 segmentation with geometric prompts.

**Pros**
- Potentially stronger text grounding in some cases

**Cons**
- Highest complexity/risk
- Out of current requested scope

## Recommended Approach
**Approach 1**: Build a clean, native SAM3 Label Studio backend class directly.

---

## Proposed Design

### 1) Backend contract and files
Target SAM3 backend directory:
- `label_studio_ml/sam3/model.py` (new backend implementation)
- Existing runtime files retained/adapted as needed (`Dockerfile`, `docker-compose.yml`, `requirements*.txt`, `start.sh`)

### 2) Runtime model lifecycle
- Initialize backend class extending `LabelStudioMLBase`
- Load/cached SAM3 model + processor once (or lazy-load on first prediction)
- Set `model_version` in `setup()`

### 3) Prompt extraction and normalization
From `context['result']`, parse:
- **Text prompt**: `TextArea` value (`value.text[0]`)
- **Keypoints**: convert from percent to absolute pixel coordinates; capture positive/negative labels
- **Rectangle**: convert percent values to pixel-space box coordinates

### 4) Prompt routing (priority strategy)
If multiple prompt types are present, select exactly one by priority:
1. `RectangleLabels`
2. `KeyPointLabels`
3. `TextArea`

Lower-priority prompt data is ignored when higher-priority prompt data exists.

### 5) Inference behavior
- Resolve image from task data via `get_local_path`
- Set image in SAM3 processor
- Apply only selected prompt mode:
  - Box mode → geometric box prompt path
  - Keypoint mode → point prompt path
  - Text mode → text prompt path
- Take top mask result (or deterministic top-k rule if needed)

### 6) Output formatting
Convert predicted mask(s) to Label Studio result JSON:
- `type: "brushlabels"`
- RLE via `brush.mask2rle`
- Include dimensions, score, and `model_version`
- Return through `ModelResponse(predictions=[...])`

### 7) Failure/edge handling
- Missing `context` or empty prompt data → empty predictions
- Invalid prompt geometry → skip invalid prompt payload
- Image read/model inference failure → log and return empty predictions

---

## Validation Plan (for implementation phase)
1. **Implement backend skeleton and parsing** → verify: backend starts and `/` health endpoint is UP.
2. **Add prompt-specific inference paths** → verify: text-only, keypoint-only, box-only produce brush predictions.
3. **Add priority routing logic** → verify: mixed prompt payload selects box > keypoint > text.
4. **Add output formatting checks** → verify: returned predictions are valid `BrushLabels` with RLE.
5. **Run focused tests** → verify: scenario-based tests for all prompt modes and empty-context behavior.

---

## Out of Scope (this design)
- Training (`fit`) logic
- Multi-tenant project-specific model isolation
- Advanced multi-prompt fusion (union/ensemble) behavior
- Video mode support

## Open Assumptions
- SAM3 inference APIs for box/point/text are available in installed package as suggested by current `sam3.py` demo usage.
- Labeling config includes required control/object tags for interactive prompting and brush output.
