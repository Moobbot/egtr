import argparse
import os
import torch
from glob import glob
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrForObjectDetection,
)
from data.visual_genome import VGDetection
from data.open_image import OIDetection
from util.misc import collate_fn


def load_id2label_from_config(config_path):
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}


def find_latest_ckpt(ckpt_dir):
    ckpts = sorted(
        glob(os.path.join(ckpt_dir, "epoch=*.ckpt")),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


class DetrForEval(torch.nn.Module):
    def __init__(
        self,
        ckpt_path,
        config_path,
        architecture="SenseTime/deformable-detr",
        device="cuda",
        strict=False,
    ):
        super().__init__()
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

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

        missing_keys, unexpected_keys = self.model.load_state_dict(
            cleaned, strict=strict
        )
        if missing_keys:
            print(
                f"[WARN] Missing keys: {missing_keys[:5]} ... ({len(missing_keys)} total)"
            )
        if unexpected_keys:
            print(
                f"[WARN] Unexpected keys: {unexpected_keys[:5]} ... ({len(unexpected_keys)} total)"
            )

        self.model.to(self.device)
        self.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to model checkpoint folder (contains 'checkpoints/' and 'config.json')",
    )
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    seed_everything(42, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load label mapping
    id2label = load_id2label_from_config(os.path.join(args.ckpt_dir, "config.json"))

    # === Load feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        "SenseTime/deformable-detr", size=800, max_size=1333
    )

    # === Load model config
    config = DeformableDetrConfig.from_pretrained(args.ckpt_dir)
    config.architecture = args.architecture
    config.num_labels = max(id2label.keys()) + 1
    config.num_queries = args.num_queries
    config.output_attention_states = False

    # === Load model
    ckpt_path = find_latest_ckpt(os.path.join(args.ckpt_dir, "checkpoints"))
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        if k.startswith("model."):
            state_dict[k[6:]] = state_dict.pop(k)

    model = DetrForEval(
        ckpt_path=ckpt_path,
        config_path=args.artifact_path,
        architecture=args.architecture,
        device="cuda",
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # === Load dataset
    if "viet_sgg" in args.data_path:
        test_dataset = VGDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )
    else:
        test_dataset = OIDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # === Logger
    logger = TensorBoardLogger(save_dir=args.ckpt_dir, name="eval")

    # === Run evaluation
    trainer = Trainer(
        precision=args.precision,
        logger=logger,
        gpus=1,
        max_epochs=-1,
    )

    print("✅ Running evaluation...")
    trainer.test(model, dataloaders=test_dataloader)
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()

# python evaluate_detr.py --data_path viet_sgg --architecture deformable-detr --ckpt_dir output/pretrain/pretrained_detr__deformable-detr/batch__8__epochs__150_50__lr__1e-05_0.0001__memo__finetune/version_0/ --split val
