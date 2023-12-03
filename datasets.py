import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted
import torch

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for
            training or testing
        transforms (None): a list of PyTorch transforms to apply to images
            and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset
            that contains the desired labels to load
        classes (None): a list of class strings that are used to define the
            mapping between class names and indices. If None, it will use
            all classes present in the given fiftyone_dataset.
    """

    def __init__(
            self,
            fiftyone_dataset,
            transforms=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.init_img_paths = self.samples.values("filepath")
        self.init_check()


    def init_check(self):
        print('checking......')
        self.img_paths = []
        for img_path in self.init_img_paths:
            img = Image.open(img_path).convert("RGB")
            try:
                img = self.transforms(img)
                self.img_paths.append(img_path)
            except:
                print(img_path, 'will not be loaded')

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.img_paths)


class ImageNetDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for
            training or testing
        transforms (None): a list of PyTorch transforms to apply to images
            and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset
            that contains the desired labels to load
        classes (None): a list of class strings that are used to define the
            mapping between class names and indices. If None, it will use
            all classes present in the given fiftyone_dataset.
    """

    def __init__(
            self,
            img_paths,
            transforms=None,
    ):
        self.transforms = transforms
        self.img_paths=img_paths


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.img_paths)

def get_imagenet_dataset(dataset, transform, transform_val):
    img_paths = []
    init_img_paths = dataset.values("filepath")
    i = 0
    for img_path in init_img_paths:
        img = Image.open(img_path).convert("RGB")

        try:
            img = transform(img)
            img_paths.append(img_path)
        except:
            i += 1
            print(img_path, 'will not be loaded')
    print(f'{i} images has been removed')
    tra_img_paths = img_paths[:800]
    val_img_paths = img_paths[800:]
    tra_dataset = ImageNetDataset(tra_img_paths, transform)
    val_dataset = ImageNetDataset(val_img_paths, transform_val)
    return tra_dataset, val_dataset

class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))



    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index])
            img = to_rgb(img)
            item = self.transform(img)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)



class Pub_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode

        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.pub_TRAIN_PATH + "/*." + c.pub_format_train)))
        else:
            # test
            self.files = sorted(glob.glob(c.pub_VAL_PATH + "/*." + c.pub_format_val))

    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index])
            img = to_rgb(img)
            item = self.transform(img)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)

transform = T.Compose([
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])




def get_dataset(name):
    if name == 'DIV':
        trainloader = DataLoader(
            Hinet_Dataset(transforms_=transform, mode="train"),
            batch_size=c.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            drop_last=True,
        )
        testloader = DataLoader(
            Hinet_Dataset(transforms_=transform_val, mode="val"),
            batch_size=c.batchsize_val,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )

    elif name == 'pub':
        trainloader = DataLoader(
            Pub_Dataset(transforms_=transform, mode="train"),
            batch_size=c.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )
        testloader = DataLoader(
            Pub_Dataset(transforms_=transform_val, mode="val"),
            batch_size=c.batchsize_val,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )

    elif name == 'COCO':
        import fiftyone.zoo as foz
        traindataset = foz.load_zoo_dataset(
            "coco-2017",
            split="test",
            max_samples=5000
        )
        testdataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            max_samples=1000
        )
        trainloader = DataLoader(
            FiftyOneTorchDataset(traindataset, transform),
            batch_size=c.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )
        testloader = DataLoader(
            FiftyOneTorchDataset(testdataset, transform_val),
            batch_size=c.batchsize_val,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )

    elif name == 'ImageNet':
        import fiftyone.zoo as foz
        dataset = foz.load_zoo_dataset(
            "imagenet-sample",
        )
        tra_dataset, val_dataset = get_imagenet_dataset(dataset, transform, transform_val)
        trainloader = DataLoader(
            tra_dataset,
            batch_size=c.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )
        testloader = DataLoader(
            val_dataset,
            batch_size=c.batchsize_val,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )
    return trainloader, testloader

if __name__ == '__main__':
    trainloader, testloader = get_dataset('COCO')
    for j in range(5):
        import torch
        img = []
        for data in trainloader:
            img.append(data)
        print()
        img = torch.cat(img, dim=0)
        print(torch.mean(img), torch.std(img))






#0.4402, 0.0793
#0.4259, 0.0773
#0.4407, 0.0799
#before DWT 0.4295, 0.2767 | tensor(0.4377) tensor(0.2797)
#after DWT
#before DWT val tensor(0.4302) tensor(0.2768)

# div 0.43, 0.28
# pub 0.904, 0.215
# ImageNet 0.434, 0.275
# COCO 0.44, 0.28