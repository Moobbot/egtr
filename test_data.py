import argparse
import json
from glob import glob

from data.hico_det import HICODetDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.visual_genome import VGDataset
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from train_egtr import collate_fn, evaluate_batch

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="viet_sgg")
parser.add_argument("--min_size", type=int, default=800)
parser.add_argument("--max_size", type=int, default=1333)
parser.add_argument("--max_topk", type=int, default=10)
parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")

args = parser.parse_args()
data_path = args.data_path
min_size = args.min_size
max_size = args.max_size
max_topk = args.max_topk
architecture = args.architecture

# Feature extractor
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    args.architecture, size=args.min_size, max_size=args.max_size
)

# test_dataset = HICODetDataset(
#     data_folder="/workspace/data/hicodet",
#     feature_extractor=feature_extractor,
#     split='test',
#     num_object_queries=200,
#     debug=True,
# )

test_dataset = VGDataset(
    data_folder="viet_sgg",
    feature_extractor=feature_extractor,
    split='train',
    num_object_queries=200,
    debug=True,
)
test_dataloader = DataLoader(
    test_dataset,
    collate_fn=lambda x: collate_fn(x, feature_extractor),
    batch_size=1,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)

# import IPython; IPython.embed()