import glob
import random
import shutil
from zipfile import ZipFile
import cv2
import pathlib

import pandas as pd
import pytorch_lightning as pt
import requests
import scipy.io as sio
import tifffile as tiff
import torch
import tqdm
import numpy as np
import os

from .patch_extractor import PatchExtractor
from .utils import rm_n_mkdir, recur_find_ext, remap_label, cropping_center
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from skimage.transform import warp, AffineTransform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2



class ConicData(Dataset):
    workdir = os.getcwd()

    def __init__(self, ids: list, download: bool,
                 from_source: bool = False, from_ome_tiff: bool = True, apply_trans=False):
        super(ConicData, self).__init__()

        self.ids = ids
        self.imgs = []
        self.labels = []
        self.download_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/download/"
        self.img_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/OME-TIFFs/"
        self.np_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/patches/"
        self.names = pd.read_csv(self.np_dir + "patch_info.csv")
        self.apply_trans = apply_trans
        self.transform = self.apply_transformation() if self.apply_trans else None
        #self.count = pd.read_csv(self.np_dir + "counts.csv")

        if len(glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/OME-TIFFs/*")) < 2:
            if download:
                self.from_zenodo()
            elif from_source:
                try:
                    self.from_source()
                except:
                    print("Could not download from source. Downloading from zenodo instead.")
                    self.from_zenodo()

        if not self._check_exists:
            raise RuntimeError("Dataset not found!")

        if from_ome_tiff:
            for idx in self.ids:
                name = self.names.iloc[idx, 0]
                img_path = os.path.join(self.img_dir, name + ".ome.tif")
                img_raw = tiff.imread(img_path)
                img = np.array(img_raw)[0:3, :, :]
                label = np.array(img_raw)[3, :, :]
                for num in np.unique(label):
                    if num > 6:
                        print(f"Warning: Found unexpected label value {num} in image {name}")
                        continue
                self.imgs.append(img)
                self.labels.append(label)
        else:
            imgs = np.load(self.np_dir + "images.npy")
            labels = np.load(self.np_dir + "labels.npy")[:, :, :, 1]
            for pair in zip(imgs, labels):
                self.imgs.append(torch.tensor(pair[0]))
                self.labels.append(torch.tensor(pair[1]))


    def __len__(self):
        return len(self.imgs)


    def apply_transformation(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
            A.OneOf([
                # Use ranges for number/size of holes
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                                hole_width_range=(0.1, 0.25), p=1.0),
                # Use ratio and unit size range for grid
                A.GridDropout(ratio=0.5, unit_size_range=(25, 51), p=1.0)
            ], p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])


    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]

        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if self.apply_trans:
            img = img.astype(np.float32)
            target = target.astype(np.int64)
            augmented = self.transform(image=img, mask=target)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = torch.tensor(img, dtype=torch.float)
            mask = torch.tensor(target, dtype=torch.long)

        if img.ndim == 3 and img.shape[2] in [1, 3]:
            img = np.transpose(img, (2, 0, 1))

        pair = (img, mask)

        return pair


    def from_zenodo(self):
        """Method for downloading the already patched and masked ome.tiff files from zenodo"""

        url = "https://zenodo.org/records/7508237/files/OME-TIFFs.zip?download=1"
        zip_path = os.path.join(self.download_dir, "OME-TIFFs.zip")
        print("Starting download...")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_size)
                        print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/total_size:.2%}", end="")

        print("\nDownload completed!")

        self._unzip_files(zip_path)
        os.remove(zip_path)
        print("Files unzipped and zip file removed.")


    def from_source(self):
        """Method for downloading, unzipping, patching and creating segmentation masked ome.tiff"""
        self._download_files()
        for zip_file in glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/download/img*"):
            self._unzip_files(zip_file)
            os.remove(zip_file)
        for file in glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/Lizard_I*/*"):
            file_name = file.split("/")[-1]
            shutil.copyfile(file, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/images/", file_name))
            os.remove(file)
        self._create_patches()
        self._numpy_to_ome_tiff()


    @staticmethod
    def _check_exists():
        if len(glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/patches/*")) > 0:
            return True
        else:
            return False


    @staticmethod
    def _download_files():
        download = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/download/"
        """Method to download the raw Lizard Dataset"""
        image_region1 = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images1.zip")
        print("Image region 1 has been downloaded")
        image_region2 = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images2.zip")
        print("Image region 2 has been downloaded")
        labels = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_labels.zip")
        print("Labels have been downloaded")
        open(download + "img1.zip", "wb").write(image_region1.content)
        open(download + "img2.zip", "wb").write(image_region2.content)
        open(download + "img_labels.zip", "wb").write(labels.content)


    @staticmethod
    def _unzip_files(zip_file):
        print(f"Unzipping file: {zip_file}")
        """General Method to unzip folders"""
        with ZipFile(zip_file, "r") as zfile:
            zfile.extractall(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/")
            zfile.close()


    @staticmethod
    def _create_patches(step_size: int = 256):
        """Patching taken from https://github.com/TissueImageAnalytics/CoNIC/blob/main/extract_patches.py

        step_size:  range of overlaps, decrease this to have a larger overlap between patches
        img_dir: should not be changed, where the finished images are
        """
        win_size = 256  # should keep this the same!
        step_size = step_size  # decrease this to have a larger overlap between patches
        extract_type = "valid"
        img_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/images/"
        ann_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/Lizard_Labels/Labels/"
        out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/patches/"

        rm_n_mkdir(out_dir)

        xtractor = PatchExtractor(win_size, step_size)

        file_path_list = recur_find_ext(img_dir, ".png")

        pbar_format = (
            "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        )
        pbar = tqdm.tqdm(
            total=len(file_path_list), bar_format=pbar_format, ascii=True, position=0
        )

        img_list = []
        inst_map_list = []
        class_map_list = []
        nuclei_counts_list = []
        patch_names_list = []

        for file_idx, file_path in enumerate(file_path_list):
            basename = pathlib.Path(file_path).stem

            img = cv2.imread(img_dir + basename + ".png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ann_load = sio.loadmat(ann_dir + basename + ".mat")
            ann_inst = ann_load["inst_map"]
            inst_class = np.squeeze(ann_load["class"]).tolist()
            inst_id = np.squeeze(ann_load["id"]).tolist()
            ann_class = np.full(ann_inst.shape[:2], 0, dtype=np.uint8)

            for val in inst_id:
                ann_inst_tmp = ann_inst == val
                idx_tmp = inst_id.index(val)
                ann_class[ann_inst_tmp] = inst_class[idx_tmp]

            ann = np.dstack([ann_inst, ann_class])

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            for idx, patch in enumerate(sub_patches):
                patch_img = patch[..., :3]  # RGB image
                patch_inst = patch[..., 3]  # instance map
                patch_class = patch[..., 4]  # class map

                # ensure nuclei range from 0 to N (N is the number of nuclei in the patch)
                patch_inst = remap_label(patch_inst)

                # only consider nuclei for counting if it exists within the central 224x224 region
                patch_inst_crop = cropping_center(patch_inst, [224, 224])
                patch_class_crop = cropping_center(patch_class, [224, 224])
                nuclei_counts_perclass = []

                # get the counts per class
                for nuc_val in range(1, 7):
                    patch_class_crop_tmp = patch_class_crop == nuc_val
                    patch_inst_crop_tmp = patch_inst_crop * patch_class_crop_tmp
                    nr_nuclei = len(np.unique(patch_inst_crop_tmp).tolist()[1:])
                    nuclei_counts_perclass.append(nr_nuclei)

                img_list.append(patch_img)
                inst_map_list.append(patch_inst)
                class_map_list.append(patch_class)
                nuclei_counts_list.append(nuclei_counts_perclass)
                patch_names_list.append("%s-%04d" % (basename, idx))

                assert patch.shape[0] == win_size
                assert patch.shape[1] == win_size

            pbar.update()
        pbar.close()

        # convert to numpy array
        img_array = np.array(img_list).astype("uint8")
        inst_map_array = np.array(inst_map_list).astype("uint16")
        class_map_array = np.array(class_map_list).astype("uint16")
        nuclei_counts_array = np.array(nuclei_counts_list).astype("uint16")

        # combine instance map and classification map to form single array
        inst_map_array = np.expand_dims(inst_map_array, -1)
        class_map_array = np.expand_dims(class_map_array, -1)
        labels_array = np.concatenate((inst_map_array, class_map_array), axis=-1)

        # convert to pandas dataframe
        nuclei_counts_df = pd.DataFrame(
            data={
                "neutrophil": nuclei_counts_array[:, 0],
                "epithelial": nuclei_counts_array[:, 1],
                "lymphocyte": nuclei_counts_array[:, 2],
                "plasma": nuclei_counts_array[:, 3],
                "eosinophil": nuclei_counts_array[:, 4],
                "connective": nuclei_counts_array[:, 5],
            }
        )
        patch_names_df = pd.DataFrame(data={"patch_info": patch_names_list})

        # save output
        np.save(out_dir + "images.npy", img_array)
        np.save(out_dir + "labels.npy", labels_array)
        nuclei_counts_df.to_csv(out_dir + "counts.csv", index=False)
        patch_names_df.to_csv(out_dir + "patch_info.csv", index=False)


    def _numpy_to_ome_tiff(self):
        data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/patches/"
        labels = np.load(data_folder + "labels.npy")
        images = np.load(data_folder + "images.npy")
        segmentations = labels[:, :, :, 0]
        classifications = labels[:, :, :, 1]
        info = pd.read_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/patches/patch_info.csv")
        for ids in range(len(images)):
            image = images[ids]
            classification = classifications[ids]
            full_image = np.zeros((256, 256, 4))
            full_image[:, :, 0] = image[:, :, 0]
            full_image[:, :, 1] = image[:, :, 1]
            full_image[:, :, 2] = image[:, :, 2]
            full_image[:, :, 3] = classification[:, :]

            full_image = np.transpose(full_image, (2, 0, 1))
            with tiff.TiffWriter(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/OME-TIFFs/", info.iloc[ids, 0] + ".ome.tif"),
                                 bigtiff=True) as tif_file:
                tif_file.write(full_image, photometric="rgb")



class ConicDataModule(pt.LightningDataModule):

    def __init__(self, **kwargs):
        super(ConicDataModule, self).__init__()
        self.workdir = os.getcwd()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs
        img_ids = list(range(0, 4981))

        self.train_ids, val_test_ids = train_test_split(img_ids, test_size=0.3, random_state=42)
        self.val_ids, self.test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=42)
        self.setup()
        self.prepare_data()

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        self.df_train = ConicData(self.train_ids, apply_trans=False, download=self.args["download"], from_source=self.args["from_source"])
        self.df_val = ConicData(self.val_ids, apply_trans=False, download=self.args["download"], from_source=self.args["from_source"])
        self.df_test = ConicData(self.test_ids, apply_trans=False, download=self.args["download"], from_source=self.args["from_source"])

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(self.df_train, batch_size=self.args['training_batch_size'], num_workers=self.args['num_workers'],
                          shuffle=True, pin_memory=True)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'],
                          shuffle=False, pin_memory=True)

    def val_dataloader(self):
        """
        :return: output - Val data loader for the given input
        """
        return DataLoader(self.df_val, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'],
                          shuffle=False, pin_memory=True)

