from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split
from collections import OrderedDict

# TODO: convert paths relative to project root
# Base paths
data_dir_path = Path("./data/raw/cvat annotations/Fruit Label 1")
annotations_path = data_dir_path.joinpath("annotations")
images_path = data_dir_path.joinpath("images")

# Load annotations
with annotations_path.joinpath(annotations_path, "instances_default.json").open() as f:
    data = json.load(f)

# Get all image ids
all_image_ids = [image["id"] for image in data["images"]]
all_annotated_img_ids = [ann["image_id"] for ann in data["annotations"]]

all_annotated_unique_img_ids = list(OrderedDict.fromkeys(all_annotated_img_ids))

# Split the data into train, validation, and test sets
train_image_ids, test_image_ids = train_test_split(
    all_annotated_unique_img_ids, test_size=0.1, random_state=42
)
train_image_ids, val_image_ids = train_test_split(
    train_image_ids, test_size=0.22, random_state=42
)  # 0.22 x 0.9 = 0.2

# Create new directories
new_dirs = {"train": train_image_ids, "val": val_image_ids, "test": test_image_ids}

dest_dir_path = Path(r".\lemon_grading\data\processed")

for new_dir, ids in new_dirs.items():
    dest_dir_path.joinpath("coco_dataset", "images", new_dir).mkdir(
        exist_ok=True, parents=True
    )
    for img_id in ids:
        filename = [
            image["file_name"] for image in data["images"] if image["id"] == img_id
        ][0]
        if dest_dir_path.joinpath("coco_dataset", "images", new_dir, filename).exists():
            pass
        else:
            shutil.copy2(
                images_path.joinpath(filename),
                dest_dir_path.joinpath("coco_dataset", "images", new_dir, filename),
            )

# Create new annotation files
for new_dir, ids in new_dirs.items():
    new_data = {
        "images": [image for image in data["images"] if image["id"] in ids],
        "annotations": [
            annotation
            for annotation in data["annotations"]
            if annotation["image_id"] in ids
        ],
        "categories": data["categories"],
    }
    if not dest_dir_path.joinpath("coco_dataset", "annotations").exists():
        dest_dir_path.joinpath("coco_dataset", "annotations").mkdir(
            exist_ok=True, parents=True
        )

    with open(
        dest_dir_path.joinpath(
            "coco_dataset", "annotations", f"instances_{new_dir}.json"
        ),
        "w",
    ) as f:
        json.dump(new_data, f)
