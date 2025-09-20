from glob import glob
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import cv2
import numpy as np

from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrForObjectDetection,
)


def find_latest_ckpt(artifact_dir: str) -> str:
    # Search for lightning checkpoints under artifact_dir (recursively)
    patterns = [
        os.path.join(artifact_dir, "**", "epoch=*.ckpt"),
        os.path.join(artifact_dir, "*.ckpt"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob(p, recursive=True))
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt files found under '{artifact_dir}'. Expected Lightning checkpoints like 'epoch=XX-*.ckpt'."
        )

    # Prefer those containing 'epoch=' and pick the largest epoch, else pick last by name
    def parse_epoch(path: str) -> int:
        base = os.path.basename(path)
        if "epoch=" in base:
            try:
                return int(base.split("epoch=")[1].split("-")[0])
            except Exception:
                return -1
        return -1

    candidates.sort(key=lambda x: (parse_epoch(x), x))
    return candidates[-1]


def load_labels_from_dataset(data_path: str):
    obj_labels = None
    val_json = os.path.join(data_path, "val.json")
    if os.path.exists(val_json):
        try:
            with open(val_json, "r", encoding="utf-8") as f:
                val = json.load(f)
            categories = val.get("categories")
            if (
                isinstance(categories, list)
                and categories
                and isinstance(categories[0], dict)
            ):
                obj_labels = [
                    c.get("name", str(c.get("id", i))) for i, c in enumerate(categories)
                ]
        except Exception:
            pass
    return obj_labels


def visualize_and_save(
    image_path: str,
    outputs: dict,
    labels: Optional[list],
    out_dir: str,
    score_thresh: float,
):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    boxes = outputs["boxes"].astype(np.int32)
    scores = outputs["scores"].astype(np.float32)
    classes = outputs["labels"].astype(np.int32)

    keep = scores >= score_thresh
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    vis = img.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x0, y0, x1, y1 = box.tolist()
        color = (
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
        )
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        name = labels[cls] if labels and 0 <= cls < len(labels) else str(cls)
        text = f"{name}:{score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x0, max(0, y0 - th - baseline)), (x0 + tw, y0), color, -1)
        cv2.putText(
            vis, text, (x0, y0 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_img = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(out_img, vis)

    # Save JSON next to image
    out_json = os.path.splitext(out_img)[0] + ".json"
    to_save = {
        "image": os.path.basename(image_path),
        "detections": [
            {
                "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                "score": float(s),
                "label_id": int(c),
                "label": (labels[c] if labels and 0 <= c < len(labels) else str(c)),
            }
            for b, s, c in zip(boxes, scores, classes)
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="DETR object detection inference (DeformableDetr)"
    )
    parser.add_argument(
        "--artifact_path",
        type=str,
        required=True,
        help="Directory containing Lightning .ckpt (and optionally config.json)",
    )
    parser.add_argument(
        "--img_folder_path", type=str, required=True, help="Folder with input images"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs/detections",
        help="Folder to save visualizations and JSON",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Optional dataset dir to read class names from val.json",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="SenseTime/deformable-detr",
        help="Base HF architecture to initialize from",
    )
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.min_size, max_size=args.max_size
    )

    # Load config: prefer artifact_path if it contains a config.json
    try:
        config = DeformableDetrConfig.from_pretrained(args.artifact_path)
    except Exception:
        config = DeformableDetrConfig.from_pretrained(args.architecture)
    # Ensure architecture stored
    config.architecture = getattr(config, "architecture", args.architecture)

    # Build model from base weights, then load fine-tuned Lightning checkpoint
    model = DeformableDetrForObjectDetection.from_pretrained(
        args.architecture, config=config, ignore_mismatched_sizes=True
    )
    ckpt_path = find_latest_ckpt(args.artifact_path)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    # Strip leading "model." added by Lightning
    for k in list(state_dict.keys()):
        if k.startswith("model."):
            state_dict[k[6:]] = state_dict.pop(k)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing 10): {missing[:10]}")
    if unexpected:
        print(
            f"[WARN] Unexpected keys: {len(unexpected)} (showing 10): {unexpected[:10]}"
        )

    model.to(device)
    model.eval()

    # Optional labels for visualization
    labels = load_labels_from_dataset(args.data_path) if args.data_path else None

    # Collect images (common extensions)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    img_paths = []
    for ext in exts:
        img_paths.extend(glob(os.path.join(args.img_folder_path, ext)))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found in '{args.img_folder_path}'.")

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        enc = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(
                pixel_values=enc["pixel_values"].to(device),
                pixel_mask=enc["pixel_mask"].to(device),
            )
        # Post-process to absolute boxes
        h, w = image.size[1], image.size[0]
        target_sizes = torch.tensor([[h, w]], device=device)
        processed = feature_extractor.post_process(outputs, target_sizes)[0]
        # Convert tensors to numpy
        det = {
            "scores": processed["scores"].detach().cpu().numpy(),
            "labels": processed["labels"].detach().cpu().numpy(),
            "boxes": processed["boxes"].detach().cpu().numpy(),
        }
        visualize_and_save(img_path, det, labels, args.output_folder, args.conf_thresh)

    print(f"Done. Results are saved to '{args.output_folder}'.")


if __name__ == "__main__":
    main()

# import torch

# print("Torch CUDA available:", torch.cuda.is_available())
# print("GPU count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))


# python detect_detr.py --artifact_path outputs\pretrain\pretrained_detr__deformable-detr\batch__8__epochs__150_50__lr__1e-05_0.0001__memo__finetune\last.ckpt --img_folder_path dataset\test --data_path dataset\val.json --architecture facebook-detr-resnet-50
