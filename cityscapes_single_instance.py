# Adapted from https://github.com/meetshah1995/pytorch-semseg
import os
import json
import torch
import numpy as np
from imageio import imwrite, imread
import random

from torch.utils import data
import torchvision.transforms.functional as tf

from utils import recursive_glob, get_boundary_map, distance_transform
from augmentation import *

class CityscapesSingleInstanceDataset(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        train_transform=Compose([RandomHorizontallyFlip(0.5), RandomRotate(10)]),
        scale_transform=Compose([Resize([224, 224])]),
        version="cityscapes",
        out_dir=""
    ):
        """__init__
        :param root: Location of data
        :param split: What split of data
        :param is_transform:
        :param img_size:
        :param augmentations: Do we or do we not want to perform augmentation
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.train_transform = train_transform
        self.scale_transform = scale_transform
        
        self.n_classes = 8  # 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        # self.images_base = os.path.join(self.root, self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        img_paths = recursive_glob(rootdir=self.images_base, suffix=".png")
        
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1,7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,]
        self.valid_classes = [
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(8)))


        self.img_paths, self.labels_coords, self.img_index_of_label, self.ins_ids = self._prepare_labels(img_paths, out_dir)
        
        if not self.img_paths:
            raise FileExistsError(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.img_paths), split))

    def load_dataset_info(self):
        info_dir = os.path.join(self.root, 'info.json')
        with open(info_dir) as f:
            info = json.load(f)
        return info
        
    def _prepare_labels(self, img_paths, out_dir):
        json_path = "./model_outputs/{}_cityscapes_single_instance_info.json".format(self.split)
        print("Save Path for JSON: ", json_path)
        if not os.path.exists(json_path):
            print("No bbox info found. Preparing labels might take some time.")
            labels_coords = []
            valid_img_paths = []
            img_index_of_label = []
            ins_ids = []
            for i, img_path in enumerate(img_paths):
                print('{}/{}'.format(i, len(img_paths)))
                img_path = img_path.rstrip()
                lbl_path = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
                )
                ins_path = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png",
                )

                lbl = imread(lbl_path)
                lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

                ins = imread(ins_path)
                ins = self.encode_insmap(np.array(ins, dtype=np.uint16), lbl)

                instances_coords = self._get_instances_coords(lbl, ins)
                if len(instances_coords) > 0:
                    valid_img_paths += [img_path]
                    labels_coords += [i[0] for i in instances_coords]
                    img_index_of_label += [len(valid_img_paths) - 1] * len(instances_coords)
                    ins_ids += [i[1] for i in instances_coords]

            with open(json_path, 'w') as f:
                json.dump({'valid_img_paths': valid_img_paths, 'labels_coords': labels_coords, 'img_index_of_label': img_index_of_label, 'ins_ids': ins_ids}, f)
                print('Saved bboxes to local.')
        else:
            with open(json_path) as f:
                json_file = json.load(f)
            valid_img_paths = json_file['valid_img_paths']
            labels_coords = json_file['labels_coords']
            img_index_of_label = json_file['img_index_of_label']
            ins_ids = json_file['ins_ids']
            
        return valid_img_paths, labels_coords, img_index_of_label, ins_ids
        
    def _get_instances_coords(self, lbl, ins):
        """
        Returns coordinates of bounding boxes of notably-large objects
        :param lbl: Semantic SegMap
        :param ins: Instance SegMap
        :return:
        """
        instances = np.unique(ins).tolist()  # Unique pixel values in instance segmap
        instances = [i for i in instances if i != 0]
        
        instances_coords = []
        for ins_num in instances:  # Iterating over unique pixel values in instance segmap
            x1, x2, y1, y2, ins_bmp = self.get_bbox(ins, ins_num)
            # filter out bbox with extreme sizes and irregular shapes
            area = np.sum(ins_bmp)
            if (area >= 100):  # If the object is large enough to be "significant"
                instances_coords += [([x1, x2, y1, y2], ins_num)]
#               occupy_ratio = np.sum(ins_bmp) / ((x2 - x1) * (y2 - y1))
            
#             if (x2 - x1 >= 50 and y2 - y1 >= 50) and (x2 - x1 <= 1000 and y2 - y1 <= 1000) \
#                and occupy_ratio > 0.25:
#                 instances_coords += [([x1, x2, y1, y2], ins_num)]
        
        return instances_coords
        
    def __len__(self):
        """__len__"""
        return len(self.labels_coords)

    def __getitem__(self, index):
        """
        Returns the training image, label which dataloader iterates through
        __getitem__
        :param index:
        """
        img_path = self.img_paths[self.img_index_of_label[index]]
        
        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        ins_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png",
        )

        lbl = imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))  # Semantic Seg Map

        ins = imread(ins_path)
        ins = self.encode_insmap(np.array(ins, dtype=np.uint16), lbl)  # Instance Segmentation Map
        
        bbox = self.labels_coords[index]
        ins[ins != self.ins_ids[index]] = 0
        ins[ins == self.ins_ids[index]] = 1
        
        img, ins = self.crop_bbox(img, ins, bbox, random_crop=(self.split=='train'), save_tag=img_path[:len(img_path) - 4])  # Ignore file extension
        
        img = Image.fromarray(img)
        ins = Image.fromarray(ins)
        if self.split == 'train':
            img, [ins] = self.train_transform(img, [ins])
        
        img, [ins] = self.scale_transform(img, [ins])
        ins_boundary = get_boundary_map(ins)

        img = tf.to_tensor(img).float()
        ins_boundary = tf.to_tensor(ins_boundary).long().squeeze(0)  # Squeeze channel dim for validation
        ins = (tf.to_tensor((np.asarray(ins) > 0).astype(np.float32)))

        return img, ins, ins_boundary, bbox
    

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def encode_insmap(self, ins, lbl):
        """

        :param ins:
        :param lbl:
        :return:
        """
        ins += 1
        ins[lbl == self.ignore_index] = 0
        instances = [i for i in np.sort(np.unique(ins)) if i != 0]
        
        for i in range(len(instances)):
            ins[ins == instances[i]] = i + 1
        return ins.astype(np.uint8)
    
    def crop_bbox(self, img, lbl, bbox, context_lo=0.1, context_hi=0.2, random_crop=True, save_tag=None):
        # assumes imgs have the same size in the first two dimensions
        H, W, _ = img.shape
        x1, x2, y1, y2 = bbox
        
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        if random_crop:
            factor = 1 + context_lo + np.random.random() * (context_hi - context_lo)
        else:
            factor = 1 + (context_lo + context_hi) / 2.
        w = (x2-x1)*factor
        h = (y2-y1)*factor
        l = max(w, h)
        x1, x2 = int(cx-l/2), int(cx+l/2)
        y1, y2 = int(cy-l/2), int(cy+l/2)
        x1, x2 = max(0, x1), min(x2, W)
        y1, y2 = max(0, y1), min(y2, H)
        
        patch_w = max(y2-y1, x2-x1)
        img_out = np.zeros((patch_w, patch_w, 3))
        lbl_out = np.zeros((patch_w, patch_w))
        
        img_out[:y2-y1, :x2-x1, :] = img[y1:y2,x1:x2,:]
        lbl_out[:y2-y1, :x2-x1] = lbl[y1:y2,x1:x2]

        # Visualization Code
        # if save_tag is not None:
        #     if random.random() < 1e-3:  # Only save small fraction of images
        #         x = save_tag.split('/')[len(save_tag) - 1]
        #         imwrite("./visualization_crops/{}_gt.png".format(x), img_out)
        #         imwrite("./visualization_crops/{}_pred.png".format(x), lbl_out)
        # End Visualization Code
        return img_out.astype(np.uint8), lbl_out.astype(np.uint8)
    
    def get_bbox(self, ins, ins_id):
        """
        Returns coordinates of bounding box around object
        :param ins:
        :param ins_id:
        :return: Returns coords
        """
        # get instance bitmap
        ins_bmp = np.zeros_like(ins)
        ins_bmp[ins == ins_id] = 1
        row_sums = ins_bmp.sum(axis=0)
        col_sums = ins_bmp.sum(axis=1)
        col_occupied = row_sums.nonzero()
        row_occupied = col_sums.nonzero()
        x1 = int(np.min(col_occupied))
        x2 = int(np.max(col_occupied))
        y1 = int(np.min(row_occupied))
        y2 = int(np.max(row_occupied))
        area = (x2 - x1) * (y2 - y1)
        return x1, x2+1, y1, y2+1, ins_bmp
    
