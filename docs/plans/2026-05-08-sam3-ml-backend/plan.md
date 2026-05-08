# SAM3 ML Backend for Label Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-ready Label Studio ML backend in `label_studio_ml/sam3` that supports `TextArea`, `KeyPointLabels`, and `RectangleLabels` prompts, with priority routing `RectangleLabels > KeyPointLabels > TextArea`, returning `BrushLabels` predictions.

**Architecture:** Add a native `LabelStudioMLBase` backend (`model.py`) that loads SAM3 once, parses interactive context payloads, selects one prompt mode by priority, runs SAM3 inference, and converts output masks to Label Studio RLE brush predictions. Keep runtime entrypoint and Docker workflow aligned with existing ML examples and add focused API tests for each prompt mode and routing behavior.

**Tech Stack:** Python, `label-studio-ml`, PyTorch, SAM3 (`sam3` package), Pillow/OpenCV, pytest.

---

## Decision Gates (must be resolved before implementation)

1. **SAM3 point-prompt API confirmation**
   - Confirm exact SAM3 method signature for keypoint prompting (single vs multiple points, positive/negative encoding) before coding Phase 2 Task 2.
2. **Labeling config contract**
   - Confirm final label config uses `BrushLabels` output and includes both smart controls (`KeyPointLabels`, `RectangleLabels`) plus `TextArea` prompt field.
3. **Top-k output policy**
   - Confirm output policy as **top-1 mask only** (current design assumption) or adjust to top-k before Phase 2 Task 3.

## Phases

1. [Phase 1: Backend scaffold and runtime wiring](phase-1.md)
2. [Phase 2: Prompt parsing, priority routing, and SAM3 inference](phase-2.md)
3. [Phase 3: API tests for prompt modes and routing](phase-3.md)
4. [Phase 4: Packaging and usage documentation](phase-4.md)
