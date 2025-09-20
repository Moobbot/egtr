import argparse
import os
import torch
from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrForObjectDetection,
)
from data.visual_genome import VGDetection
from data.open_image import OIDetection
from lib.evaluation.coco_eval import CocoEvaluator
from util.misc import collate_fn


def load_id2label_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}


def find_latest_ckpt(ckpt_dir):
    ckpts = sorted(glob(os.path.join(ckpt_dir, "epoch=*.ckpt")), key=lambda x: int(x.split("epoch=")[1].split("-")[0]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


class DetrForEval(torch.nn.Module):
    def __init__(self, ckpt_path, config_path, architecture="SenseTime/deformable-detr", device="cuda", strict=False):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Load config
        config = DeformableDetrConfig.from_pretrained(config_path)
        config.architecture = architecture
        config.output_attention_states = False
        self.model = DeformableDetrForObjectDetection(config=config)

        # Load state_dict from Lightning .ckpt
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[6:]] = v
            else:
                cleaned[k] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(cleaned, strict=strict)
        if missing_keys:
            print(f"[WARN] Missing keys: {missing_keys[:5]} ... ({len(missing_keys)} total)")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys: {unexpected_keys[:5]} ... ({len(unexpected_keys)} total)")

        self.model.to(self.device)
        self.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@torch.no_grad()
def run_eval(model, dataloader, feature_extractor, coco_evaluator, device):
    model.eval()
    predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([t["orig_size"] for t in labels], dim=0).to(device)
        results = feature_extractor.post_process(outputs, orig_target_sizes)

        # Eval with COCO API
        res = {
            target["image_id"].item(): output
            for target, output in zip(labels, results)
        }
        coco_evaluator.update(res)

        # Save predictions
        for target, output in zip(labels, results):
            image_id = target["image_id"].item()
            boxes = output["boxes"].tolist()
            scores = output["scores"].tolist()
            labels_ = output["labels"].tolist()
            predictions.append({
                "image_id": image_id,
                "boxes": boxes,
                "scores": scores,
                "labels": labels_
            })

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    ap50 = coco_evaluator.coco_eval["bbox"].stats[1]

    return predictions, ap50

def collate_fn_with_extractor(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = load_id2label_from_config(os.path.join(args.ckpt_dir, "config.json"))

    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )

    ckpt_path = find_latest_ckpt(os.path.join(args.ckpt_dir, "checkpoints"))
    model = DetrForEval(
        ckpt_path=ckpt_path,
        config_path=args.ckpt_dir,
        architecture=args.architecture,
        device=device
    )

    # Dataset
    if "viet_sgg" in args.data_path or "visual_genome" in args.data_path:
        dataset = VGDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )
    else:
        dataset = OIDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=lambda x: collate_fn_with_extractor(x, feature_extractor),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # COCO Eval
    coco_evaluator = CocoEvaluator(dataset.coco, ["bbox"])

    print("✅ Running evaluation...")
    predictions, ap50 = run_eval(model, dataloader, feature_extractor, coco_evaluator, device)
    print(f"✅ AP50: {ap50:.3f}")

    # Save results
    output_name = f"{ckpt_path.replace('.ckpt', '')}__{args.split}__results.json"
    with open(output_name, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"✅ Predictions saved to: {output_name}")


if __name__ == "__main__":
    main()


# python evaluate_detr.py --data_path viet_sgg --architecture deformable-detr --ckpt_dir output/pretrain/pretrained_detr__deformable-detr/batch__8__epochs__150_50__lr__1e-05_0.0001__memo__finetune/version_0/ --split val
