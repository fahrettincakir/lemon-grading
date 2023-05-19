import src.utils.data_utils as data_utils
import numpy as np
from imgaug.augmentables.polys import Polygon
import cv2 as cv
import imgaug.augmenters as iaa


class ImageData:
    def __init__(self, img_id, coco, data_dir, data_type, classes, augmenter):
        self.img_id = img_id
        self.coco = coco
        self.classes = classes
        self.binary_masks_color = 1
        self.colored_masks_color = (0, 255, 255)

        # img_array
        self.img_array = data_utils.load_img(img_id, coco, data_dir, data_type)
        self.img_array_augmented = []

        # binary_masks
        self.binary_masks = [
            np.zeros(
                (self.img_array.shape[0], self.img_array.shape[1], 1), dtype=np.uint8
            )
            for _ in range(len(self.classes))
        ]
        self.binary_masks_augmented = []

        # colored_masks
        self.colored_masks = [
            np.ones(
                (
                    self.img_array.shape[0],
                    self.img_array.shape[1],
                    self.img_array.shape[2],
                )
            )
            * 255
            for _ in range(len(self.classes))
        ]
        self.colored_masks_augmented = []

        # semantic_masks
        self.semantic_masks = []
        self.semantic_masks_augmented = []

        # instance_masks
        self.instance_masks = []
        self.instance_masks_augmented = []

        # bounding_box_masks
        self.bounding_box_masks = []
        self.bounding_box_masks_augmented = []

        # keypoint_masks
        self.keypoint_masks = []
        self.keypoint_masks_augmented = []

        # alpha_masks
        self.alpha_masks = []
        self.alpha_masks_augmented = []

        self.augmenter = augmenter

        self.process_image(print_summary=True, augment=True)

    def add_mask(self, mask_data):
        # Add a new mask to the image data
        # ...
        pass

    def remove_mask(self, mask_id):
        # Remove a specific mask from the image data
        # ...
        pass

    def modify_mask(self, mask_id, new_mask_data):
        # Modify a specific mask with new data
        # ...
        pass

    def get_polygons_annotation(self):
        anns = data_utils.get_annotations(self.coco, self.classes)
        img_dict = data_utils.imgID_2_annotations(anns)

        polygons_list = []
        polygons_vertices_list = []
        category_idx_list = []

        for ann in img_dict[self.img_id]:
            category_name = self.coco.loadCats(ann["category_id"])[0]["name"]
            category_idx = self.classes.index(category_name)

            segmentation = ann["segmentation"]

            for polygon in segmentation:
                polygon = [tuple(point) for point in np.reshape(polygon, (-1, 2))]
                polygons_vertices_list.append(polygon)
                polygons_list.append(Polygon(polygon))

                category_idx_list.append(category_idx)

        return (polygons_list, polygons_vertices_list, category_idx_list)

    def fill_regions_of_interest(
        self,
        polygons_list,
        polygons_vertices_list,
        category_idx_list,
    ):
        filled_binary_masks = [mask.copy() for mask in self.binary_masks]
        filled_colored_masks = [mask.copy() for mask in self.colored_masks]

        polygons_classes = tuple(zip(polygons_list, category_idx_list))
        i = 0
        for polygon, class_idx in polygons_classes:
            polygon_array_inalist = [
                np.array(polygons_vertices_list[i]).astype(np.int32)
            ]
            cv.fillPoly(
                filled_binary_masks[class_idx],
                polygon_array_inalist,
                self.binary_masks_color,
            )

            cv.fillPoly(
                filled_colored_masks[class_idx],
                polygon_array_inalist,
                self.colored_masks_color,
            )
            i = i + 1

        return filled_binary_masks, filled_colored_masks

    def process_image(self, print_summary, augment=True):
        color_blacks_masks = self.binary_masks_color
        color_white_masks = self.colored_masks_color

        (
            polygons_list,
            polygons_vertices_list,
            category_idx_list,
        ) = self.get_polygons_annotation()

        self.binary_masks, self.colored_masks = self.fill_regions_of_interest(
            polygons_list,
            polygons_vertices_list,
            category_idx_list,
        )

        self.binary_masks = np.stack(self.binary_masks, axis=-1)
        self.colored_masks = np.stack(self.colored_masks, axis=-1)

        if print_summary:
            # Get summary information
            num_classes = self.binary_masks.shape[-1]
            height, width = self.binary_masks.shape[:2]
            num_pixels = height * width
            unique_values = np.unique(self.binary_masks)
            num_unique_values = len(np.unique(self.binary_masks))
            min_value = np.min(self.binary_masks)
            max_value = np.max(self.binary_masks)

            # Print summary information
            print("filled_binary_masks shape:", self.binary_masks.shape)
            print("Number of classes:", num_classes)
            print("Number of channels (each mask):", self.binary_masks.shape[2])
            print("Height:", height)
            print("Width:", width)
            print("Number of pixels:", num_pixels)
            print("Unique values:", unique_values)
            print("Number of unique values:", num_unique_values)
            print("Minimum value:", min_value)
            print("Maximum value:", max_value)

        if augment:
            self.img_array_augmented = self.augmenter(image=self.img_array)

            a = [
                self.augmenter(image=np.squeeze(x, axis=3))
                for x in np.split(
                    self.binary_masks, self.binary_masks.shape[-1], axis=-1
                )
            ]

            self.binary_masks_augmented = np.stack(a, axis=-1)

            b = [
                self.augmenter(image=np.squeeze(x, axis=3))
                for x in np.split(
                    self.colored_masks, self.colored_masks.shape[-1], axis=-1
                )
            ]

            self.colored_masks_augmented = np.stack(b, axis=-1)


if __name__ == "__main__":
    print(1)
