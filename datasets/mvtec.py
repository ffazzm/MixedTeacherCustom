import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='../dataset/MVTEC', class_name='bottle', is_train=True,
                 resize=224, cropsize=224):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_modified_dataset_folder()

        # self.transform_x = T.Compose([T.Resize(resize),
        #                               T.CenterCrop(cropsize),
        #                               T.ToTensor(),
        #                               T.Normalize(mean=[0.485, 0.456, 0.406],
        #                                           std=[0.229, 0.224, 0.225])])

        self.transform_x = T.Compose([T.Resize(resize,resize),
                                    #   T.CenterCrop(cropsize),
                                    #   T.RandomCrop(cropsize),
                                    #   T.RandomHorizontalFlip(p=0.5),
                                    #   T.RandomVerticalFlip(p=0.2),
                                      T.ColorJitter(brightness=0.2, contrast=0.2),
                                    #   T.RandomRotation(degrees=15),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        
        self.transform_x_val = T.Compose([T.Resize(resize,resize),
                                    #   T.CenterCrop(cropsize),
                                    #   T.RandomCrop(cropsize),
                                    #   T.RandomHorizontalFlip(p=0.5),
                                    #   T.RandomVerticalFlip(p=0.2),
                                    #   T.ColorJitter(brightness=0.2, contrast=0.2),
                                    #   T.RandomRotation(degrees=15),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
                                                
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x_path, y, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        # Load and transform the input image
        x = Image.open(x_path).convert('RGB')
        if self.is_train:
            x = self.transform_x(x)
        else:
            x = self.transform_x_val(x)

        # Load the mask, or create blank mask if not available
        if mask_path is None or not os.path.isfile(mask_path):
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_modified_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        gt_train_dir = os.path.join(self.dataset_path, self.class_name, 'train_ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

                # Determine the correct ground truth directory
                gt_type_dir = os.path.join(gt_dir if phase == 'test' else gt_train_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]

                # Build ground truth mask paths, allow for missing files
                gt_fpath_list = []
                for img_fname in img_fname_list:
                    mask_path = os.path.join(gt_type_dir, img_fname + '_mask.png')
                    if os.path.isfile(mask_path):
                        gt_fpath_list.append(mask_path)
                    else:
                        # If mask file does not exist, will generate blank mask later
                        gt_fpath_list.append(None)

                mask.extend(gt_fpath_list)

        assert len(x) == len(y) == len(mask), 'Mismatch in x, y, and mask counts'

        return list(x), list(y), list(mask)
