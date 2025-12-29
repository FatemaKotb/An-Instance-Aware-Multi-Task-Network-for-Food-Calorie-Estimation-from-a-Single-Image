import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional


FOOD101_ROOT_DEFAULT = Path(os.environ.get("FOOD101_ROOT", "/kaggle/input/food-101/food-101"))
SUBSET_META_DEFAULT = Path("food101_cache_k25/subset_meta.json")
FUSION_CKPT_DEFAULT = "fusion_head_k25.pt"


def _extract_pred_label_from_text(text: str) -> Optional[str]:
    """
    Extract the last occurrence of a raw line that starts with `PRED_LABEL=...`.
    """
    for line in reversed(text.splitlines()):
        if line.startswith("PRED_LABEL="):
            return line.split("=", 1)[1].strip()
    return None


def _build_inprocess_predictor() -> Callable[..., str]:
    """
    Run the pipeline in-process (no subprocess per image).

    Preference:
      1) Use a direct callable (predict_one / run_one / predict / run_pipeline) if it exists.
      2) Fallback to calling scripts.run_pipeline.main() with a temporary sys.argv.

    Logging:
      - We fully suppress pipeline stdout/stderr to keep evaluator logs clean.
      - We still parse `PRED_LABEL=...` from the captured text.
    """
    try:
        import scripts.run_pipeline as rp  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import scripts.run_pipeline. Run from the project root "
            "so `scripts/` is importable."
        ) from e

    # 1) Direct callable if present (best case).
    for name in ("predict_one", "run_one", "run_pipeline_one", "predict", "run_pipeline"):
        fn = getattr(rp, name, None)
        if callable(fn):

            def _predict_direct(
                *,
                image_path: Path,
                pipeline_mode: str,
                fusion_ckpt: Optional[str],
                topk: int,
            ) -> str:
                out = fn(
                    image=str(image_path),
                    pipeline_mode=pipeline_mode,
                    fusion_ckpt=fusion_ckpt,
                    topk=topk,
                )
                if isinstance(out, str):
                    return out

                if isinstance(out, dict):
                    for k in ("pred_label", "label", "food_class_name"):
                        v = out.get(k, None)
                        if v is not None:
                            return str(v)

                for attr in ("pred_label", "label", "food_class_name"):
                    if hasattr(out, attr):
                        v = getattr(out, attr)
                        if v is not None:
                            return str(v)

                raise RuntimeError(
                    f"{name}() did not return a usable label. "
                    "Either update this evaluator to match its return type or ensure it returns a label."
                )

            return _predict_direct

    # 2) Fallback: call rp.main() and parse PRED_LABEL from captured output.
    main_fn = getattr(rp, "main", None)
    if not callable(main_fn):
        raise RuntimeError(
            "scripts.run_pipeline does not expose a callable `main()` or any known direct prediction function."
        )

    def _predict_via_main(
        *,
        image_path: Path,
        pipeline_mode: str,
        fusion_ckpt: Optional[str],
        topk: int,
    ) -> str:
        argv_backup = sys.argv[:]
        buf = io.StringIO()

        sys.argv = [
            "run_pipeline",
            "--image",
            str(image_path),
            "--pipeline_mode",
            pipeline_mode,
            "--topk",
            str(topk),
        ]
        if pipeline_mode == "fusion_head" and fusion_ckpt:
            sys.argv += ["--fusion_ckpt", fusion_ckpt]

        # Suppress ALL pipeline output (stdout + stderr) for clean evaluator logs.
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            main_fn()

        sys.argv = argv_backup
        text = buf.getvalue()

        pred = _extract_pred_label_from_text(text)
        if pred is None:
            tail = text[-1200:] if len(text) > 1200 else text
            raise RuntimeError(
                "Could not parse prediction. Ensure the pipeline prints a raw line like `PRED_LABEL=<class>`.\n"
                f"Last captured output:\n{tail}"
            )
        return pred

    return _predict_via_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--food101_root", type=str, default=str(FOOD101_ROOT_DEFAULT))
    p.add_argument("--subset_meta", type=str, default=str(SUBSET_META_DEFAULT))
    p.add_argument("--fusion_ckpt", type=str, default=FUSION_CKPT_DEFAULT)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="If > 0, evaluate only the first N filtered test images.")
    p.add_argument("--print_every", type=int, default=10, help="Print running accuracy every N images (per mode).")
    p.add_argument(
        "--log_steps",
        action="store_true",
        help="If set, prints per-image step lines (start/end) in addition to running accuracy.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    food101_root = Path(args.food101_root)
    subset_meta = Path(args.subset_meta)
    fusion_ckpt = args.fusion_ckpt
    topk = max(1, int(args.topk))

    meta = json.loads(subset_meta.read_text())
    chosen = set(meta["chosen_class_names"])  # 25 class names

    test_txt = food101_root / "meta" / "test.txt"
    images_dir = food101_root / "images"

    rels = [line.strip() for line in test_txt.read_text().splitlines() if line.strip()]
    rels = [r for r in rels if r.split("/", 1)[0] in chosen]
    if args.limit and args.limit > 0:
        rels = rels[: args.limit]

    predict = _build_inprocess_predictor()

    results = {}
    for mode in ["hybrid_clip", "fusion_head"]:
        correct = 0
        total = 0
        t0 = time.time()

        for r in rels:
            cls, img_id = r.split("/")
            img_path = images_dir / cls / f"{img_id}.jpg"

            if args.log_steps:
                print(f"[{mode}] START {total+1}/{len(rels)}  img={cls}/{img_id}.jpg")

            pred = predict(
                image_path=img_path,
                pipeline_mode=mode,
                fusion_ckpt=fusion_ckpt,
                topk=topk,
            )

            correct += int(pred == cls)
            total += 1

            if args.log_steps:
                status = "OK" if pred == cls else "MISS"
                print(f"[{mode}] END   {total}/{len(rels)}  pred={pred}  true={cls}  {status}")

            if args.print_every > 0 and (total % args.print_every == 0 or total == len(rels)):
                elapsed = time.time() - t0
                ips = total / max(elapsed, 1e-9)
                print(f"[{mode}] {total}/{len(rels)}  acc={100*correct/max(total,1):.2f}%  ({ips:.2f} img/s)")

        acc = 100.0 * correct / max(total, 1)
        results[mode] = acc
        print(f"{mode}: {acc:.2f}%  (n={total})")

    print("\nFill Table 1 with:")
    print(f"Pretrained CLIP: {results['hybrid_clip']:.2f}")
    print(f"Instance-aware fusion (ours): {results['fusion_head']:.2f}")


if __name__ == "__main__":
    main()
