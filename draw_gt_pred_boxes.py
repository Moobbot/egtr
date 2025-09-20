import os
import json
from PIL import Image, ImageDraw, ImageFont

def draw_gt_and_pred_boxes(image_path, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, id2label, save_path, score_thresh=0.3):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # === GT: xanh lá
    for box, label_id in zip(gt_boxes, gt_labels):
        x0, y0, w, h = box  # coco-format GT: [x, y, width, height]
        x1, y1 = x0 + w, y0 + h
        label = id2label.get(label_id, str(label_id))
        draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
        draw.text((x0 + 2, y0 + 2), f"GT: {label}", fill="green", font=font)

    # === Pred: đỏ
    for box, label_id, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_thresh:
            continue
        x0, y0, x1, y1 = map(int, box)
        label = id2label.get(label_id, str(label_id))
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0 + 2, y0 + 2), f"{label}: {score:.2f}", fill="red", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def load_annotations(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id → file_name
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # image_id → list of annotations
    gt_map = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        bbox = ann["bbox"]
        label = ann["category_id"]
        gt_map.setdefault(img_id, []).append((bbox, label))

    return id2file, gt_map


def load_predictions(pred_json_path):
    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # image_id → predictions
    pred_map = {}
    for p in preds:
        img_id = p["image_id"]
        boxes = p["boxes"]
        labels = p["labels"]
        scores = p["scores"]
        pred_map[img_id] = {"boxes": boxes, "labels": labels, "scores": scores}

    return pred_map


def main():
    # === Đường dẫn
    coco_path = "viet_sgg/val.json"
    pred_path = "output/pretrain/pretrained_detr__deformable-detr/batch__8__epochs__150_50__lr__1e-05_0.0001__memo__finetune/version_0/checkpoints/epoch=01-validation_loss=16.74__val__results.json"
    image_dir = "viet_sgg/coco_uitvic_test"  # hoặc val
    output_dir = "vis_gt_pred"
    score_thresh = 0.3

    # === id2label
    id2label = {
        0: "bóng chày", 1: "bảng tỉ số", 2: "giày", 3: "găng bóng chày", 4: "gậy bóng chày",
        5: "huấn luyện viên", 6: "khung thành", 7: "khán giả", 8: "lưới", 9: "quả bóng bầu dục",
        10: "quả bóng chuyền", 11: "quả bóng chày", 12: "quả bóng rổ", 13: "quả bóng tennis",
        14: "quả bóng đá", 15: "rổ bóng rổ", 16: "sân bóng chuyền", 17: "sân bóng chày",
        18: "sân bóng đá", 19: "sân tennis", 20: "trẻ em", 21: "trọng tài",
        22: "vận động viên", 23: "vợt tennis", 24: "đồng phục"
    }

    id2file, gt_map = load_annotations(coco_path)
    pred_map = load_predictions(pred_path)

    # === Duyệt từng ảnh
    for image_id in sorted(gt_map.keys()):
        file_name = id2file[image_id]
        image_path = os.path.join(image_dir, file_name)
        save_path = os.path.join(output_dir, file_name)

        # GT
        gt_boxes, gt_labels = zip(*gt_map[image_id]) if image_id in gt_map else ([], [])

        # Pred
        if image_id in pred_map:
            pred = pred_map[image_id]
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]
        else:
            pred_boxes, pred_labels, pred_scores = [], [], []

        draw_gt_and_pred_boxes(
            image_path,
            gt_boxes,
            gt_labels,
            pred_boxes,
            pred_labels,
            pred_scores,
            id2label,
            save_path,
            score_thresh=score_thresh
        )

    print("✅ Saved visualizations to:", output_dir)


if __name__ == "__main__":
    main()
