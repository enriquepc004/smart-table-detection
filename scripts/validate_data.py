
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "roboflow_v1"

CLASSES = ["fork", "glass", "knife", "napkin", "plate"]
SPLITS = ["train", "valid", "test"]

total_images = 0
total_boxes = 0
errors = []
video_splits = {}

for split in SPLITS:
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    label_files = list(labels_dir.glob("*.txt"))
    boxes = 0
    empty = 0

    for image in images:
        label = labels_dir / f"{image.stem}.txt"

        # Check that the image has a label file
        if not label.exists():
            errors.append(f"Missing label: {image.name}")
            continue

        # Extract video ID from original naming convention
        filename = image.stem
        if "_IMG_" in filename:
            source = filename.split("_IMG_")[0] + "_IMG_" + (
                filename.split("_IMG_")[1].split("_")[0]
            )
            video_splits.setdefault(source, set()).add(split)

        lines = label.read_text(encoding="utf-8").splitlines()

        if not lines:
            empty += 1

        for line in lines:
            values = line.split()

            if len(values) != 5:
                errors.append(f"Invalid format: {label.name}")
                continue

            try:
                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])
            except ValueError:
                errors.append(f"Invalid values: {label.name}")
                continue

            if not 0 <= class_id < len(CLASSES):
                errors.append(f"Invalid class: {label.name}")

            if not (0 <= x <= 1 and 0 <= y <= 1
                    and 0 < w <= 1 and 0 < h <= 1):
                errors.append(f"Invalid box: {label.name}")
            elif (x - w / 2 < -1e-4 or
                  x + w / 2 > 1.0001 or
                  y - h / 2 < -1e-4 or
                  y + h / 2 > 1.0001):
                errors.append(f"Box outside image: {label.name}")

            boxes += 1

    image_stems = {p.stem for p in images}
    for label in label_files:
        if label.stem not in image_stems:
            errors.append(f"Orphan label: {label.name}")

    total_images += len(images)
    total_boxes += boxes

    print(f"{split}: {len(images)} images, "
          f"{boxes} boxes, {empty} empty labels")

for video, splits in video_splits.items():
    if len(splits) > 1:
        errors.append(f"Video leakage: {video} -> {splits}")

print(f"\nTotal images: {total_images}")
print(f"Total bounding boxes: {total_boxes}")
print(f"Potential errors: {len(errors)}")

for error in errors[:30]:
    print("ERROR:", error)

if not errors:
    print("\nDataset validation PASSED!")
