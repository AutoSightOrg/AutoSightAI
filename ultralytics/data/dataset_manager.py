import torch
from torchvision.transforms import v2
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
from torchvision import tv_tensors
from tqdm import tqdm


class DatasetManager:
    def __init__(
        self,
        root_dir,
        class_names,
        transform=None,
        DEBUG=False,
    ):
        # directories
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "images")
        self.label_dir = os.path.join(self.root_dir, "labels")
        self.transformed_image_dir = os.path.join(self.root_dir, "transformed_images")
        self.transformed_label_dir = os.path.join(self.root_dir, "transformed_labels")
        os.makedirs(self.transformed_image_dir, exist_ok=True)
        os.makedirs(self.transformed_label_dir, exist_ok=True)

        # init
        self.image_size_wh = (640, 480)
        self.image_size_hw = (480, 640)

        self.transform = transform

        self.class_names = class_names
        self.images_by_classes, self.classes_by_images = self._init_data()

        # debug
        self.DEBUG = DEBUG

    def _init_data(self):
        images_by_classes = [[] for i in range(len(self.class_names))]
        classes_by_images = {}
        for file in os.listdir(self.label_dir):
            if file.endswith(".txt"):
                with open(os.path.join(self.label_dir, file), "r") as label_file:
                    img_name = file.replace(".txt", ".jpg")
                    classes_by_images[img_name] = []
                    for line in label_file:
                        class_ = int(line.split(" ")[0])
                        images_by_classes[class_].append(img_name)
                        classes_by_images[img_name].append(class_)

        return images_by_classes, classes_by_images

    def print_files_count_by_class(self):
        for i, images in enumerate(self.images_by_classes):
            print(f"{i} : {len(images)}")

    def plot_class_distribution(self):
        class_counts = [len(images) for images in self.images_by_classes]

        colors = [plt.cm.viridis(random.random()) for _ in range(len(class_counts))]

        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(class_counts)),
            class_counts,
            align="center",
            alpha=1.0,
            color=colors,
        )
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.title("Classes Distribution Map")

        plt.xticks(range(len(class_counts)), self.class_names, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    def get_class_fraction_images(self, class_fraction):
        class_counts = [len(images) for images in self.images_by_classes]
        class_counts_needed = [
            max(round(count * class_fraction), 1) for count in class_counts
        ]

        selected_images = []
        for cls, images in enumerate(self.images_by_classes):
            for image in images:
                if class_counts_needed[cls] > 0:
                    added = True
                    for cls2 in self.classes_by_images[image]:
                        if class_counts_needed[cls2] <= 0:
                            added = False
                        class_counts_needed[cls2] -= 1
                    if not added:
                        for cls2 in self.classes_by_images[image]:
                            class_counts_needed[cls2] += 1
                    else:
                        selected_images.append(image)
                else:
                    break
            while class_counts_needed[cls] > 0:
                random_img = random.choice(self.images_by_classes[cls])
                for cls2 in self.classes_by_images[random_img]:
                    class_counts_needed[cls2] -= 1
                selected_images.append(random_img)

        return selected_images

    def dataset_balancing(self):
        images_to_oversample = self.get_images_to_oversample()
        self._transform_images_balance(images_to_oversample, 0)

    def get_images_to_oversample(self):
        # needed instance counts from each class for oversampling
        class_counts = [
            len(images_by_class) for images_by_class in self.images_by_classes
        ]
        max_class_count = max(class_counts)
        class_counts_needed = [
            max_class_count - count if count != 0 else 0 for count in class_counts
        ]

        images_to_oversample = (
            []
        )  # array to store image names which are going to be oversampled

        # oversampling (main loop)
        for class_ in range(len(class_counts_needed)):
            current_class_images = self.images_by_classes[class_].copy()
            improved = True
            while class_counts_needed[class_] > 0 and improved:
                improved = False
                random.shuffle(current_class_images)  # increase randomity
                for img in current_class_images:
                    add = True
                    for class2_ in self.classes_by_images[img]:
                        if class_counts_needed[class2_] <= 0:
                            add = False
                            break
                    if add:
                        for class2_ in self.classes_by_images[img]:
                            class_counts_needed[class2_] -= 1
                        improved = True
                        images_to_oversample.append(img)

        # add images to the remained minority classes (even if it creates a slight inbalance)
        for class_ in range(len(class_counts_needed)):
            while class_counts_needed[class_] > 0:
                random_img = random.choice(self.images_by_classes[class_])
                for class2_ in self.classes_by_images[random_img]:
                    class_counts_needed[class2_] -= 1
                images_to_oversample.append(random_img)

        return images_to_oversample

    def _transform_images_balance(self, images_to_oversample, start_nr):
        images_to_transform_numbering = {}
        for image_name in images_to_oversample:
            images_to_transform_numbering[image_name] = -1 + start_nr

        self._write(
            len(images_to_oversample),
            images_to_oversample,
            images_to_transform_numbering,
            balance=True,
        )

    def transform_images_by_class(self, class_number, images_count, start_nr):
        if class_number < 0 or class_number >= len(self.class_names):
            print("Invalid class number.")
            return

        images_to_transform = self.images_by_classes[class_number]

        images_to_transform_numbering = {}
        for image_name in images_to_transform:
            images_to_transform_numbering[image_name] = -1 + start_nr

        self._write(
            images_count,
            images_to_transform,
            images_to_transform_numbering,
        )

    def _write(
        self,
        images_count,
        images_to_transform,
        images_to_transform_numbering,
        balance=False,
    ):
        for i in tqdm(range(images_count), desc=f"Transforms"):
            # select file to transform
            if balance:
                image_name = images_to_transform[i]
            else:
                image_name = random.choice(images_to_transform)

            base_name, extension = os.path.splitext(image_name)

            img, yolo_bboxes = self._read_img_and_bboxes(image_name)
            if self.transform:
                transformed_img, labels_yolo_format = self.transform_image(
                    image_name, img, yolo_bboxes
                )
            else:
                # no change
                transformed_img = img
                labels_yolo_format = yolo_bboxes

            images_to_transform_numbering[image_name] += 1
            transformed_file_name = (
                f"{base_name}_{images_to_transform_numbering[image_name]}{extension}"
            )
            transformed_img_path = os.path.join(
                self.transformed_image_dir, transformed_file_name
            )
            transformed_img.save(transformed_img_path)

            with open(
                os.path.join(
                    self.transformed_label_dir,
                    transformed_file_name.replace(".jpg", ".txt"),
                ),
                "w",
            ) as transformed_label_file:
                for label in labels_yolo_format:
                    transformed_label_file.write(
                        f"{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n"
                    )

    def _read_img_and_bboxes(self, image_file_name):
        # read image
        img = Image.open(os.path.join(self.image_dir, image_file_name)).convert("RGB")

        # read labels from file
        label_file_name = image_file_name.replace(".jpg", ".txt")
        label_file_path = os.path.join(self.label_dir, label_file_name)
        with open(label_file_path, "r") as label_file:
            yolo_bboxes = [list(map(float, line.split())) for line in label_file]

        return img, yolo_bboxes

    def transform_image(self, img, yolo_bboxes):

        bboxes = tv_tensors.BoundingBoxes(
            self._yolo_to_xyxy(torch.tensor([bbox[1:] for bbox in yolo_bboxes])),
            format="XYXY",
            canvas_size=self.image_size_hw,
        )

        # transform
        transformed_img, out_bboxes = self.transform(img, bboxes)

        if self.DEBUG:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img)
            self._plot_bounding_boxes(bboxes, "red")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Transformed")
            plt.imshow(transformed_img)
            self._plot_bounding_boxes(out_bboxes, "blue")
            plt.axis("off")

            plt.show()

        labels_yolo_format = [
            torch.cat((torch.tensor(int(yolo_bboxes[i][0])).unsqueeze(0), box))
            for i, box in enumerate(self._xyxy_to_yolo(out_bboxes))
        ]

        return transformed_img, labels_yolo_format

    def _plot_bounding_boxes(self, bboxes, color):
        for bbox in bboxes:
            x, y, w, h = bbox
            rect = plt.Rectangle(
                (x, y), w - x, h - y, linewidth=2, edgecolor=color, facecolor="none"
            )
            plt.gca().add_patch(rect)

    def _yolo_to_xyxy(self, yolo_bboxes):
        x, y, w, h = (
            yolo_bboxes[:, 0],
            yolo_bboxes[:, 1],
            yolo_bboxes[:, 2],
            yolo_bboxes[:, 3],
        )
        x_min = (x - w / 2) * self.image_size_hw[1]
        y_min = (y - h / 2) * self.image_size_hw[0]
        x_max = (x + w / 2) * self.image_size_hw[1]
        y_max = (y + h / 2) * self.image_size_hw[0]
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def _xyxy_to_yolo(self, xyxy_bboxes):
        x_min, y_min, x_max, y_max = (
            xyxy_bboxes[:, 0],
            xyxy_bboxes[:, 1],
            xyxy_bboxes[:, 2],
            xyxy_bboxes[:, 3],
        )
        center_x = (x_min + x_max) / (2 * self.image_size_hw[1])
        center_y = (y_min + y_max) / (2 * self.image_size_hw[0])
        width = (x_max - x_min) / self.image_size_hw[1]
        height = (y_max - y_min) / self.image_size_hw[0]
        return torch.stack([center_x, center_y, width, height], dim=1)
