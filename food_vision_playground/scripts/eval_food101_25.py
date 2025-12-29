import json
import os
import subprocess
from pathlib import Path

FOOD101_ROOT = Path(os.environ.get("FOOD101_ROOT", "/kaggle/input/food-101/food-101"))
SUBSET_META  = Path("food101_cache_k25/subset_meta.json")
FUSION_CKPT   = "fusion_head_k25.pt"

def run_one(image_path: Path, mode: str):
    cmd = ["python", "-m", "scripts.run_pipeline", "--image", str(image_path), "--pipeline_mode", mode]
    if mode == "fusion_head":
        cmd += ["--fusion_ckpt", FUSION_CKPT]

    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    # EXPECTED: your run_pipeline logs a line like: PRED_LABEL=pizza
    pred = None
    for line in reversed(out.splitlines()):
        if line.startswith("PRED_LABEL="):
            pred = line.split("=", 1)[1].strip()
            break
    if pred is None:
        raise RuntimeError(f"Could not parse prediction. Add a 'PRED_LABEL=...' print.\nLast output:\n{out[-500:]}")
    return pred

def main():
    meta = json.loads(SUBSET_META.read_text())
    chosen = set(meta["chosen_class_names"])  # 25 class names :contentReference[oaicite:2]{index=2}

    test_txt = FOOD101_ROOT / "meta" / "test.txt"
    images_dir = FOOD101_ROOT / "images"

    # Build filtered test list
    rels = [line.strip() for line in test_txt.read_text().splitlines() if line.strip()]
    rels = [r for r in rels if r.split("/", 1)[0] in chosen]

    # Evaluate
    results = {}
    for mode in ["hybrid_clip", "fusion_head"]: # The threshold for egytian model is set at 100% so it is never chosen which makes this identical to pretrained_clip
        correct = 0
        total = 0
        for r in rels:
            cls, img_id = r.split("/")
            img_path = images_dir / cls / f"{img_id}.jpg"
            pred = run_one(img_path, mode)
            correct += int(pred == cls)
            total += 1
        acc = 100.0 * correct / max(total, 1)
        results[mode] = acc
        print(f"{mode}: {acc:.2f}%  (n={total})")

    print("\nFill Table 1 with:")
    print(f"Pretrained CLIP: {results['hybrid_clip']:.2f}")
    print(f"Instance-aware fusion (ours): {results['fusion_head']:.2f}")

if __name__ == "__main__":
    main()
