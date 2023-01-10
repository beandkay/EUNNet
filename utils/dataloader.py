import torch, numpy, torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from .utils import *
from .randaugment import RandAugmentPC
from PIL import Image

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size
    i_tensor = torch.zeros((len(imgs), 3, w, h), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = numpy.asarray(img, dtype=numpy.uint8)
        if (nump_array.ndim < 3):
            nump_array = numpy.expand_dims(nump_array, axis=-1)
        nump_array = numpy.rollaxis(nump_array, 2)
        i_tensor[i] = torch.from_numpy(nump_array.copy())
    return i_tensor.contiguous(), targets

def folder_loader(traindir, valdir, batch_size):
    jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
    lighting = Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])

    train_dataset = torchvision.datasets.ImageFolder(traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # RandAugmentPC(n=1, m=9),
            # transforms.ToTensor(),
            # jittering,
            # lighting
            ]))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, 
        num_workers=8, shuffle=True, drop_last=True, collate_fn=fast_collate)

    val_dataset = torchvision.datasets.ImageFolder(valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor()
            ]))
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size * 3, 
        num_workers=8, collate_fn=fast_collate)
    return train_loader, val_loader

