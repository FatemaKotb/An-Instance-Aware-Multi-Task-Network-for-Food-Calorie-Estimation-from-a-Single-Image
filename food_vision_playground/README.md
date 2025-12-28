# Food Vision Playground

This project runs a full pipeline for food instance segmentation, depth estimation, food type prediction, and calorie estimation from a single RGB image.

## Pipeline overview (levels)

### Level 1 — Basic signals (computed directly from the image)

#### 1A) Backbone (features)
Extracts useful internal features from the image.

#### 1B) Segmentation (instance masks)
Finds objects and returns a mask per detected instance.

#### 1C) Depth estimation
Predicts a depth value for every pixel.

### Level 2 — Fusion (currently a stub)
Combines backbone features + depth into a compact representation per instance.

### Level 3 — Prediction head (currently a stub + pretrained model)
Predicts the food label for the instance. A stub that can be replaced with a learnable module later exists but is not used here; instead, a pretrained CLIP-based head is used.

### Level 4 — Physics-based calories (rule-based)
Estimates volume and calories from geometry + food label lookup.


## How to run

```bash
cd food_vision_playground
python -m scripts.run_pipeline --image data/sample.jpg --seg_thresh 0.5
```


## What gets saved in `runs/<image_stem>/` after running

Typical outputs:

* `input.png` (the input image)
* `mask_###.png` (binary mask per instance)
* `masks_overlay.png` (all masks drawn on the image)
* `depth.npy` (depth map as a numpy array)


## What gets logged to console during run
* Detected instances + their predicted food labels + top-5 probabilities.
* Estimated volume (in cm³) and calories (in kcal) per instance.
* Total estimated calories for the image.

## Design Decisions and next steps

* The estimated depth here is relative depth (not absolute). To get any meaningful volume estimates, the model needs to be calibrated with a known reference object in the scene or any other method to convert relative depth to absolute depth.
* The fusion module and prediction head are currently stubs or pretrained models. These should be replaced with learnable modules and trained end-to-end.
* The physics-based calorie estimation relies on sample lookup tables containing very few entries for density and kcal/g values. Finding official nutrition databases with more comprehensive coverage would allow estimating calories for a wider variety of food items.
* All the development here has been done while using a simple pizza image as input. Testing and validating the pipeline on more diverse and complex food images is necessary. And I am referring to food categories the current pretrained models can recognize as a starting point; I am not considering retraining or fine-tuning any models on new food categories at this time.
* I need to know the reason we want to estimate the portion in the prediction head because from what I understand, the volume estimation is already done in the physics-based calorie estimation block. So is it redundant to have portion estimation in the prediction head as well? Or is there a different definition of "portion" that I am missing here?