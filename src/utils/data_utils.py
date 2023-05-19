from pycocotools.coco import COCO
from pathlib import Path
import imageio.v2 as imageio


def load_img(img_id, coco, data_dir, data_type):
    img_info = coco.loadImgs(img_id)[0]
    img_file = "{}/images/{}".format(data_dir, img_info["file_name"])
    img_file = str(Path(data_dir).joinpath("images", data_type, img_info["file_name"]))
    img_array = imageio.imread(img_file)
    return img_array


def imgID_2_annotations(annotations):
    # Create dictionary to map image ID to annotations
    img_dict = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_dict:
            img_dict[img_id] = []
        img_dict[img_id].append(ann)
    return img_dict


def imgID_2_filepaths(coco, data_dir, data_type):
    imgid_2_file = {}
    for image in coco.dataset["images"]:
        image_id = image["id"]
        file_name = image["file_name"]
        imgid_2_file[image_id] = str(
            Path(data_dir).joinpath("images", data_type, file_name)
        )
    return imgid_2_file


def file_2_imgid(coco):
    file_2_imgid = {}
    for image in coco.dataset["images"]:
        image_id = image["id"]
        file_name = image["file_name"]
        file_2_imgid[file_name] = image_id
    return file_2_imgid


def get_annotations(coco, classes):
    catIds = coco.getCatIds(catNms=classes)
    imgIds = coco.getImgIds(catIds=catIds)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=imgIds, catIds=catIds))
    return annotations


def get_categoryName_of_annotation(coco, ann):
    category_name = category_name = coco.loadCats(ann["category_id"])[0]["name"]
    return category_name


def get_categoryIdx_of_categoryName(classes, category_name):
    ctgryidx = classes.index(category_name)
    return ctgryidx


def get_binary_class(img_id, coco, img_dict, positive_classes):
    binary_class_value = 0

    for ann in img_dict[img_id]:
        category_name = get_categoryName_of_annotation(coco, ann)

        if category_name in positive_classes:
            binary_class_value = 1
            break

    return binary_class_value
