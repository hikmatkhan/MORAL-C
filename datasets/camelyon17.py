import random
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, hflip, vflip, affine

import numpy as np
from torchvision.transforms.functional import to_tensor


class Camelyon17DatasetWithMasks(Dataset):
    def __init__(self, root_dir, hospital: int, split, transform=None, get_mask=False,
                 geometric_aug=True, affine_aug=False):
        super(Camelyon17DatasetWithMasks, self).__init__()
        self.root_dir = Path(root_dir)
        self.hospital = hospital
        if split == 'train':
            self.split = [0]
        elif split == 'val':
            self.split = [1]
        elif split == 'all':
            self.split = [0, 1]
        else:
            raise ValueError(f'split must be "train" or "val" or "all" but received split={split} instead.')
        self.transform = transform
        self.get_mask = get_mask

        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'

        self._input_array = []
        self.y_array = []

        self._read_and_initialise_metadata()

        self.using_tar = True
        self.tar_files = {}

        self._create_input_arrays()

        self.multiply_mask_with_x_prob = 0.5
        self.geometric_aug = geometric_aug
        self.affine_aug = affine_aug

    def __len__(self):
        return len(self._input_array)

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def _read_and_initialise_metadata(self):
        df = pd.read_csv(self.root_dir / 'metadata.csv', index_col=0, dtype={'patient': 'str'})
        self.df = df[(df['center'] == self.hospital) & (df['split'].isin(self.split))]
        self.df.reset_index(drop=True, inplace=True)

        self.metadata_array = torch.stack(
            (torch.LongTensor(self.df['center'].values.astype('long')),
             torch.LongTensor(self.df['slide'].values)), dim=1
        )

    def _create_input_arrays(self):
        cols = ['patient', 'node', 'x_coord', 'y_coord', 'tumor']
        label_array = []
        for patient, node, x, y, tumor in self.df.loc[:, cols].itertuples(index=False, name=None):
            tar_filename = f'patient_{patient}_node_{node}.tar'
            patch_name = f'patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            # This tuple is same for both patches and their masks
            self._input_array.append((tar_filename, patch_name))
            label_array.append(tumor)
        self.y_array = torch.LongTensor(label_array)

    def get_input(self, idx):
        """Returns x, mask_x for a given index(idx)."""
        x = self._read_image(self.images_dir, idx)
        x = self._transform_image(x, self.transform)
        if self.get_mask:
            mask_x = self._read_image(self.masks_dir, idx)
            mask_x = self._transform_image(mask_x, None)
            # Data Augmentation: Either use binary mask times input instead
            if torch.rand(1) < self.multiply_mask_with_x_prob:
                x = mask_x * x

        if self.geometric_aug and torch.rand(1) < 0.5:
            x = hflip(x)
            if self.get_mask:
                mask_x = hflip(mask_x)

        if self.geometric_aug and torch.rand(1) < 0.5:
            x = vflip(x)
            if self.get_mask:
                mask_x = vflip(mask_x)

        if self.affine_aug:
            translate = [random.randint(-45, 45), random.randint(-45, 45)]
            angle = random.randint(-90, 90)
            x = affine(x, translate=translate, angle=angle, scale=1, shear=[0], fill=[1.])
            if self.get_mask:
                mask_x = affine(mask_x, translate=translate, angle=angle, scale=1, shear=[0], fill=[0.])

        if self.get_mask:
            return x, mask_x
        return x, x

    '''def _read_image(self, patches_dir, idx):
        """Check my blog post explaining why this: https://medium.com/p/35097f72ebbd"""
        patient_tar_filename, patch_name = self._input_array[idx]
        if str(patches_dir / patient_tar_filename) in self.tar_files:
            patient_tar = self.tar_files[str(patches_dir / patient_tar_filename)]
        else:
            patient_tar = tarfile.open(patches_dir / patient_tar_filename)
            self.tar_files[str(patches_dir / patient_tar_filename)] = patient_tar
        x = self._read_image_from_tar(patient_tar, patch_name)

        return x'''

    def _read_image(self, patches_dir, idx):
        """Check which tar file has issues while reading images."""
        patient_tar_filename, patch_name = self._input_array[idx]
        tar_path = str(patches_dir / patient_tar_filename)

        try:
            # Check if tar file is already opened
            if tar_path in self.tar_files:
                patient_tar = self.tar_files[tar_path]
            else:
                # Try opening the tar file and store it in memory
                patient_tar = tarfile.open(tar_path)
                self.tar_files[tar_path] = patient_tar

            # Try extracting the image from tar
            x = self._read_image_from_tar(patient_tar, patch_name)
            return x

        except tarfile.ReadError:
            print(f"Error: Corrupt tar file detected -> {tar_path}")
            return None
        except KeyError as e:
            print(f"Error: Missing {patch_name} in {tar_path}")
            return None
        except Exception as e:
            print(f"Unexpected error with tar file {tar_path}: {e}")
            return None

    @staticmethod
    def _read_image_from_tar(patient_tar, patch_name):
        try:
            img = Image.open(BytesIO(patient_tar.extractfile(f'./{patch_name}').read())).convert('RGB')
        except KeyError as e:
            print(f'Received KeyError for file {patch_name} and btw tar file has len {len(patient_tar.getmembers())}')
            raise e

        return img

    @staticmethod
    def _transform_image(image, transform):
        if transform is not None:
            return transform(image)
        return to_tensor(image)


class Camelyon17DatasetWithNucleiBlackened(Camelyon17DatasetWithMasks):
    def __init__(self, root_dir, hospital: int, split, transform=None, geometric_aug=True, affine_aug=False):
        super().__init__(root_dir, hospital, split, transform, get_mask=False, geometric_aug=geometric_aug, affine_aug=affine_aug)

        super(Camelyon17DatasetWithMasks, self).__init__()
        self.root_dir = Path(root_dir)
        self.hospital = hospital
        if split == 'train':
            self.split = [0]
        elif split == 'val':
            self.split = [1]
        elif split == 'all':
            self.split = [0, 1]
        else:
            raise ValueError(f'split must be "train" or "val" or "all" but received split={split} instead.')
        self.transform = transform
        self.get_mask = get_mask

        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'

        self._input_array = []
        self.y_array = []

        self._read_and_initialise_metadata()

        self.using_tar = True
        self.tar_files = {}

        self._create_input_arrays()

        self.multiply_mask_with_x_prob = 0.5
        self.geometric_aug = geometric_aug
        self.affine_aug = affine_aug

    def __len__(self):
        return len(self._input_array)

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def _read_and_initialise_metadata(self):
        df = pd.read_csv(self.root_dir / 'metadata.csv', index_col=0, dtype={'patient': 'str'})
        self.df = df[(df['center'] == self.hospital) & (df['split'].isin(self.split))]
        self.df.reset_index(drop=True, inplace=True)

        self.metadata_array = torch.stack(
            (torch.LongTensor(self.df['center'].values.astype('long')),
             torch.LongTensor(self.df['slide'].values)), dim=1
        )

    def _create_input_arrays(self):
        cols = ['patient', 'node', 'x_coord', 'y_coord', 'tumor']
        label_array = []
        for patient, node, x, y, tumor in self.df.loc[:, cols].itertuples(index=False, name=None):
            tar_filename = f'patient_{patient}_node_{node}.tar'
            patch_name = f'patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            # This tuple is same for both patches and their masks
            self._input_array.append((tar_filename, patch_name))
            label_array.append(tumor)
        self.y_array = torch.LongTensor(label_array)



    def get_input(self, idx):
        """Overrides get_input() to blacken nuclei."""
        x = self._read_image(self.images_dir, idx)  # Load original image
        x = self.blacken_nuclei(x)  # Apply nuclei blackening
        x = self._transform_image(x, self.transform)  # Transform image
        return x, x  # Returning twice to match model input-output

    def blacken_nuclei(self, image):
        """Detect and blacken nuclei in the image."""
        img_np = np.array(image)  # Convert to numpy array

        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Apply mask to blacken nuclei
        img_np[mask > 0] = [0, 0, 0]

        return Image.fromarray(img_np)  # Convert back to PIL image


    def _read_image(self, patches_dir, idx):
        """Check which tar file has issues while reading images."""
        patient_tar_filename, patch_name = self._input_array[idx]
        tar_path = str(patches_dir / patient_tar_filename)

        try:
            # Check if tar file is already opened
            if tar_path in self.tar_files:
                patient_tar = self.tar_files[tar_path]
            else:
                # Try opening the tar file and store it in memory
                patient_tar = tarfile.open(tar_path)
                self.tar_files[tar_path] = patient_tar

            # Try extracting the image from tar
            x = self._read_image_from_tar(patient_tar, patch_name)
            return x

        except tarfile.ReadError:
            print(f"Error: Corrupt tar file detected -> {tar_path}")
            return None
        except KeyError as e:
            print(f"Error: Missing {patch_name} in {tar_path}")
            return None
        except Exception as e:
            print(f"Unexpected error with tar file {tar_path}: {e}")
            return None


    @staticmethod
    def _read_image_from_tar(patient_tar, patch_name):
        try:
            img = Image.open(BytesIO(patient_tar.extractfile(f'./{patch_name}').read())).convert('RGB')
        except KeyError as e:
            print(f'Received KeyError for file {patch_name} and btw tar file has len {len(patient_tar.getmembers())}')
            raise e

        return img

    @staticmethod
    def _transform_image(image, transform):
        if transform is not None:
            return transform(image)
        return to_tensor(image)
