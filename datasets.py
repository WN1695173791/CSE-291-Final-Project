import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from guided_diffusion.guided_diffusion.image_datasets import _list_image_files_recursively
from transforms import CustomDataAugmentation
from extraction import collect_features


class ImageImageDataset(Dataset):
    '''
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = CustomDataAugmentation(min_scale=0.6)
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        # assert pil_image.size[0] == pil_image.size[1], \
        #        f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label


class ImageLabelDataset(Dataset):
    '''
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        # assert pil_image.size[0] == pil_image.size[1], \
        #        f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label

class Pair(Dataset):
    def __init__(self, opts, files, feature_extractor, train=True):
        self.opts = opts
        self.files = sorted(files)
        self.trans = CustomDataAugmentation(min_scale=0.6)
        self.train = train
        self.feature_extractor = feature_extractor

        self.val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            lambda x: 2 * x - 1
        ])


        x = torch.linspace(-1, 1, 256)
        y = torch.linspace(-1, 1, 256)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.files[idx]).convert('RGB'))
        # pos_img = np.concatenate([img, self.x_grid[:, :, None], self.y_grid[:, :, None]], axis=2)

        if self.train:
            crops_transformed, coords, flags = self.trans(img)
            img1 = crops_transformed[0].half().cuda()
            img2 = crops_transformed[1].half().cuda()
            coord1 = coords[0]
            coord2 = coords[1]
            flag1 = flags[0]
            flag2 = flags[1]

        else:
            img1 = self.val_trans(img)[None].half().cuda()
            img2 = self.val_trans(img)[None].half().cuda()

        features1 = self.feature_extractor(img1, noise=None)
        features2 = self.feature_extractor(img2, noise=None)
        x1 = collect_features(self.opts, features1).cpu()
        x2 = collect_features(self.opts, features2).cpu()

        if self.train:
            return x1, x2, coord1, coord2, flag1, flag2
        else:
            return x1, x2

