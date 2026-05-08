# Phase 4: Packaging and usage documentation

**Goal:** Ensure SAM3 backend is runnable by contributors via Docker/local flow and has minimal usage documentation with required labeling config.

**Tasks:** 2 related tasks.

### Task 1: Align dependency/runtime files with implemented backend

**Files:**
- Modify: `label_studio_ml/sam3/requirements.txt`
- Modify: `label_studio_ml/sam3/requirements-base.txt`
- Modify: `label_studio_ml/sam3/docker-compose.yml`
- Modify: `label_studio_ml/sam3/Dockerfile`

- [ ] **Step 1: Write a failing container smoke check command**

Run: `cd label_studio_ml/sam3 && docker compose config`
Expected: PASS for syntax; if missing variables/invalid keys are found, treat as failure to fix before build.

- [ ] **Step 2: Normalize runtime dependencies for SAM3 backend only**

```text
# requirements-base.txt
label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git
gunicorn==22.0.0

# requirements.txt
# keep only SAM3 + inference dependencies actually imported by model.py
```

- [ ] **Step 3: Ensure docker env variables reflect SAM3 names**

```yaml
# docker-compose.yml (environment)
- DEVICE=cuda
- MODEL_CONFIG=configs/sam3/sam3_hiera_l.yaml
- MODEL_CHECKPOINT=sam3_hiera_large.pt
```

- [ ] **Step 4: Run local static checks on runtime files**

Run: `cd label_studio_ml/sam3 && docker compose config`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add label_studio_ml/sam3/requirements.txt label_studio_ml/sam3/requirements-base.txt label_studio_ml/sam3/docker-compose.yml label_studio_ml/sam3/Dockerfile
git commit -m "chore(sam3): align packaging and runtime dependencies"
```

### Task 2: Add SAM3 backend README with labeling config and run instructions

**Files:**
- Create: `label_studio_ml/sam3/README.md`
- Reference: `label_studio_ml/examples/segment_anything_2_image/README.md`

- [ ] **Step 1: Write failing doc-presence check**

Run: `test -f label_studio_ml/sam3/README.md`
Expected: FAIL (before file creation).

- [ ] **Step 2: Add README with exact sections**

```markdown
# SAM3 Label Studio ML Backend

## Supported interactive inputs
- TextArea
- KeyPointLabels
- RectangleLabels

## Prompt priority
RectangleLabels > KeyPointLabels > TextArea

## Labeling config example
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="brush" toName="image">
    <Label value="defect" background="#FFA39E"/>
  </BrushLabels>
  <KeyPointLabels name="kp" toName="image" smart="true">
    <Label value="defect" background="#250DD3"/>
  </KeyPointLabels>
  <RectangleLabels name="rect" toName="image" smart="true">
    <Label value="defect" background="#FFC069"/>
  </RectangleLabels>
  <TextArea name="prompt" toName="image" perRegion="false"/>
</View>

## Run
- Docker
- Local

## Test
pytest -q
```

- [ ] **Step 3: Verify docs and tests command references are valid**

Run: `cd label_studio_ml/sam3 && test -f README.md && pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add label_studio_ml/sam3/README.md
git commit -m "docs(sam3): add usage guide and labeling config"
```

## Acceptance Criteria
- Runtime files are internally consistent for local/docker startup.
- `README.md` exists with supported prompt modes, priority behavior, labeling config, and run/test instructions.

## Phase Verification
Run: `cd label_studio_ml/sam3 && docker compose config && pytest -q`
Expected: PASS
