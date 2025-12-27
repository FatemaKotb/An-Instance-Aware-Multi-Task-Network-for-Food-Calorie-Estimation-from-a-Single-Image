# Food Vision Playground (Fusion + Heads Scaffold)

This repo is a clean, modular scaffold for an instance-level food pipeline:

- **Backbone** extracts multi-scale features
- **Segmenter** produces instance masks
- **Depth** predicts a dense depth map
- **Fusion** produces three explicit per-instance outputs: **Fi, fi, vi**
- **Prediction Head (stub)** consumes fusion outputs to predict **food class + portion**
- **Physics Head (stub)** consumes geometry + predicted class to compute **Area/Volume/Calories**

The goal is to keep the pipeline runnable end-to-end now, while making it easy to swap stubs with real modules later.

---

## Blocks and Interfaces (Exact I/O)

All dataclasses live in `scripts/dtypes.py`.

### 1) EfficientNetBlock (Backbone mode)
**File:** `scripts/efficientnet.py`  
**Class:** `EfficientNetBlock(mode="backbone")`

**Input**
- `img_rgb_uint8`: `np.ndarray` of shape `[H, W, 3]`, dtype `uint8`

**Output**
- `BackboneOutput`
  - `features`: `List[torch.Tensor]`
    - each tensor shape `[1, C_s, H_s, W_s]`
    - typically multiple scales (e.g., 4 tensors)

---

### 2) MaskRCNNTorchVisionBlock (Instance Segmentation)
**File:** `scripts/maskrcnn_torchvision.py`  
**Class:** `MaskRCNNTorchVisionBlock`

**Input**
- `img_rgb_uint8`: `np.ndarray [H, W, 3] uint8`
- `score_thresh`: `float` (e.g., `0.5`)

**Output**
- `InstanceSegmentationResult`
  - `boxes_xyxy`: `np.ndarray [N, 4]` (float)
  - `scores`: `np.ndarray [N]` (float)
  - `class_ids`: `np.ndarray [N]` (int) *(COCO ids)*
  - `masks`: `np.ndarray [N, H, W]` (bool)

---

### 3) ZoeDepthBlock (Monocular Depth)
**File:** `scripts/zoedepth.py`  
**Class:** `ZoeDepthBlock`

**Input**
- `img_rgb_uint8`: `np.ndarray [H, W, 3] uint8`

**Output**
- `DepthResult`
  - `depth`: `np.ndarray [H, W] float32`

---

### 4) FusionStub (Instance-Aware Fusion Stub)
**File:** `scripts/fusion_stub.py`  
**Class:** `FusionStub`

Fusion produces the **three explicit outputs** shown in the diagram:

- **Fi** = masked instance features (multi-scale)
- **fi** = global pooled feature vector
- **vi** = instance depth descriptor

**Input**
- `pooled_feats`: `List[torch.Tensor]`
  - each tensor `[C_s]` (mask-pooled from backbone feature map `s`)
  - stored on CPU
- `depth_median`: `float`
- `depth_mean`: `float`

**Output**
- `FusionOutput`
  - `masked_features` (**Fi**): `List[torch.Tensor]` (each `[C_s]`)
  - `global_features` (**fi**): `torch.Tensor [D]` (e.g., concatenation of Fi)
  - `instance_depth` (**vi**): `torch.Tensor [2]` = `[depth_median, depth_mean]`

---

### 5) Prediction Head Stub (Classification + Portion)
**File:** `scripts/prediction_head_stub.py`  
**Class:** `PredictionHeadStub`

**Input**
- `fusion`: `FusionOutput`
  - uses at minimum: `fusion.global_features` (**fi**)
  - may also use: `fusion.instance_depth` (**vi**) and/or `fusion.masked_features` (**Fi**)

**Output**
- `PredictionOutput`
  - `food_class_id`: `int` *(stub may output a placeholder)*
  - `food_confidence`: `float`
  - `portion`: `float` *(stub value; later could be grams / bins / scalar)*

> Note: in the *final* system, this head predicts food type from the fused embedding/features.
> The COCO `class_ids` from segmentation are not food labels.

---

### 6) Physics Head Stub (Area / Volume / Calories)
**File:** `scripts/physics_head_stub.py`  
**Class:** `PhysicsHeadStub`

**Input**
- `mask`: `np.ndarray [H, W] bool`
- `depth_map`: `np.ndarray [H, W] float32`
- `prediction`: `PredictionOutput` *(food class used for density lookup)*

**Output**
- `PhysicsOutput`
  - `area_px`: `int` (Aᵢ in pixel units)
  - `volume_proxy`: `float` (Vᵢ proxy, e.g., `area_px * depth_median`)
  - `calories_proxy`: `float` (Cᵢ proxy using density table / default)

---

### 7) Pipeline (Composed System)
**File:** `scripts/pipeline.py`  
**Class:** `Pipeline`

**Input**
- `img_rgb_uint8`: `np.ndarray [H, W, 3] uint8`

**Output**
- `PipelineOutput`
  - `instance_outputs`: `List[InstanceOutputs]`
  - `depth_map`: `np.ndarray [H, W] float32`
  - `backbone_features`: `List[torch.Tensor]` (raw multi-scale maps)

Each `InstanceOutputs` contains:
- instance data: `mask`, `box_xyxy`, `score`
- pooled geometry: `area_px`, `depth_median`, `depth_mean`
- structured stage outputs:
  - `fusion: FusionOutput`
  - `prediction: PredictionOutput`
  - `physics: PhysicsOutput`

---

## End-to-End Dataflow Diagram (Mermaid)

```mermaid
flowchart LR
  A["Input RGB image<br/>H x W x 3 uint8"] --> B["EfficientNetBlock<br/>mode: backbone"]
  A --> S["MaskRCNNTorchVisionBlock"]
  A --> D["ZoeDepthBlock"]

  B -->|"BackboneOutput.features<br/>List of 1 x Cs x Hs x Ws"| P["Per-instance pooling<br/>mask-pool features + depth stats"]
  S -->|"InstanceSegmentationResult<br/>masks, boxes, scores"| P
  D -->|"DepthResult.depth<br/>H x W float32"| P

  P -->|"pooled_feats + depth_median/mean"| F["FusionStub<br/>outputs Fi, fi, vi"]
  F -->|"FusionOutput"| H["PredictionHeadStub<br/>food class + portion"]
  H -->|"PredictionOutput"| X["PhysicsHeadStub<br/>area, volume, calories"]

  F --> O["PipelineOutput"]
  H --> O
  X --> O
````

To run the full pipeline on an image:

```bash
cd food_vision_playground
python -m scripts.run_pipeline --image data/sample.jpg --seg_thresh 0.5
```