# Google Colab Notebooks

This folder contains the Colab notebooks used to **collect data** and **generate annotations** for the project dataset.

* **`FoodDetectionDataCollection.ipynb`**
  Collects and organizes raw food images for different dishes/cuisines (download/scrape + basic structuring for later annotation).

* **`DINO+SAM Annotation.ipynb`**
  Generates instance-level annotations using a **DINO → SAM** pipeline (detect food regions with DINO, then refine masks with SAM). Outputs masks/metadata that can be used to build the training dataset.
