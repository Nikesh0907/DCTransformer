from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder
from dataset import DatasetFromFolder2
from torch.utils.data import DataLoader


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform():
    return Compose([
        ToTensor(),
    ])



def get_patch_training_set(upscale_factor, patch_size, root_dir=None):
    if root_dir is None:
        root_dir = "/data/CAVEdata12/"
    # X: HSI, Y: MSI, X_blur: blurred HSI
    train_dir1 = join(root_dir, "train/X")
    train_dir2 = join(root_dir, "train/Y")
    train_dir3 = join(root_dir, "train/X_blur")

    return DatasetFromFolder(train_dir1,train_dir2,train_dir3,upscale_factor, patch_size, input_transform=input_transform())


def get_test_set(root_dir=None):
    if root_dir is None:
        root_dir = "/data/CAVEdata12/"
    test_dir1 = join(root_dir, "test/X")
    test_dir2 = join(root_dir, "test/Y")
    test_dir3 = join(root_dir, "test/Z")

    return DatasetFromFolder2(test_dir1,test_dir2,test_dir3, input_transform=input_transform())


if __name__ == '__main__':
    # smoke test (only runs if dataset exists locally)
    upscale = 8
    patch = 64
    try:
        train_set = get_patch_training_set(upscale, patch)
        test_set = get_test_set()
        training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=2,
                                          shuffle=False)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1,
                                         shuffle=False)
        z, y, x = next(iter(training_data_loader))
        print('Train batch shapes:', z.shape, y.shape, x.shape)
        zt, yt, xt, name = next(iter(testing_data_loader))
        print('Test sample shapes:', zt.shape, yt.shape, xt.shape, 'name:', name[0])
    except Exception as e:
        print('Dataset not available or error:', e)

