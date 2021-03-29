import io
import os
import json
import zipfile

import PIL
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

MEANS = [0.48043839, 0.44820218, 0.39760034]
STDS = [0.27698959, 0.26908774, 0.28216029]


##########################################################################################
def split_dataset(dataset, validation_size, seed=42):
    """ Splits a dataset into a training set and a validation set
    using sklearn.model_selection.StratifiedShuffleSplit
    to generate a even split accross all different labels
    
    Args:
        dataset (torch.utils.data.Dataset):
            samples are in `dataset.data`
            labels are in `dataset.targets`
        validation_size (float): number in [0, 1]
            fraction of data to use in the validation set
        seed (int, optional):
            seed for the random number generator
    
    Returns:
        tuple (train_idx, validation_idx): numpy.ndarray
            indices of train and validation sets
    """
    shuffler = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    train_idx, validation_idx = next(iter(shuffler.split(dataset.data, dataset.targets)))
    
    return train_idx, validation_idx
    

##########################################################################################
class ACSE44Dataset(VisionDataset):
    """ ACSE-4 Dataset for The Identification Game competition
        
    Args:
        npz_file (str): path to npz binary storing the data
            data must have dtype `np.uint8`, labels must have dtype `np.int64`
            must have keys `train_data`, `train_labels` if train=True
            must have key `test_data` if train=False
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    ### ####################################
    def __init__(self, npz_file, train=True, transform=None, target_transform=None):
       
        super(ACSE44Dataset, self).__init__(os.path.split(npz_file)[0],
                                            transform=transform,
                                            target_transform=target_transform)
        self.train = train
        self.data = list()
        self.targets = list()
        self.filenames = list()
        self.class_to_idx = dict()

        # TODO: is it faster with this mmap_mode='r' or with mmap_mode=None (default) ?
        npzfile = np.load(npz_file)
        error = None

        try:
            if 'category_map' in npzfile.files:
                self.class_to_idx = {str(k): int(v) for k, v in npzfile['category_map']}
            else:
                print("Warning: No category -> index map found in npz file")
            
            if train:
                assert 'train_data' in npzfile.files, "array with key 'train_data' not found in npz file"
                assert 'train_labels' in npzfile.files, "array with key 'train_labels' not found in npz file"
                self.data = torch.from_numpy(npzfile['train_data'])
                self.targets = torch.from_numpy(npzfile['train_labels'])
                
                if 'train_filenames' in npzfile.files:
                    self.filenames = npzfile['train_filenames']
                else:
                    print("Warning: No train file names found in npz file")
            
            else:
                assert 'test_data' in npzfile.files, "array with key 'test_data' not found in npz file"
                self.data = torch.from_numpy(npzfile['test_data'])
                self.targets = torch.zeros(self.data.size(0))
                self.targets[:] = np.nan
                
                if 'test_filenames' in npzfile.files:
                    self.filenames = npzfile['test_filenames']
                else:
                    print("Warning: No test file names found in npz file")
                
        except AssertionError as e:
            error = e
        
        finally:
            npzfile.close()
            if error is not None:
                raise e

        self._to_pil = transforms.ToPILImage()


    ### ####################################
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = self._to_pil(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    
    ### ####################################
    def __len__(self):
        return len(self.data)


##########################################################################################
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


##########################################################################################
def make_npz_dump(zip_file, out_file):
    
    print("Extracting images from zip archive ...")
    train_images, train_labels, train_file_names, test_images, test_file_names, category_map = _read_zip_archive(zip_file)
    
    print("Done. Converting to numpy arrays ...")
    arrays = {
        "train_data": _make_numpy_array(train_images),
        "train_labels": np.array(train_labels).astype(np.int64),
        "train_filenames": np.array(train_file_names),
        "test_data": _make_numpy_array(test_images),
        "test_filenames": np.array(test_file_names),
        "category_map": np.array([[k, v] for k, v in category_map.items()])
    }
    
    print("Done. Saving to npz file ...")
    if os.path.exists(out_file):
        os.remove(out_file)
    np.savez(out_file, **arrays)
    
    print("Done.")
    
    return True
    

##########################################################################################
def _make_numpy_array(list_of_PIL_images):
    t = transforms.ToTensor()
    gs = transforms.Compose([transforms.Grayscale(3), t])
    
    arrays = list()
    for im in list_of_PIL_images:
        if im.mode == "RGB":
            array = (t(im).numpy() * 255).astype(np.uint8)
        elif im.mode == "L":
            array = (gs(im).numpy() * 255).astype(np.uint8)
        else:
            raise NotImplementedError("Invalid jpeg image mode '{}'".format(im.mode))
        
        arrays.append(array)
        
    return np.stack(arrays)


##########################################################################################
def _read_zip_archive(zip_file):
    train_images = list()
    train_labels = list()
    train_file_names = list()
    test_images = list()
    test_file_names = list()
    category_map = dict()
    
    with zipfile.ZipFile(zip_file) as archive:
        with archive.open("mapping.json") as mapping_file:
            category_map = json.load(io.BytesIO(mapping_file.read()))

        for name in archive.namelist():
            path = name.split("/")

            if path[-1].lower().endswith(".jpeg") or path[-1].lower().endswith(".jpg"):
                if path[0] == "test":
                    with archive.open(name) as image_file:
                        image = PIL.Image.open(io.BytesIO(image_file.read()))
                        image.load()
                        test_images.append(image)
                    test_file_names.append(path[-1].lower())

                elif path[0] == "train":
                    category = category_map.get(path[1])
                    assert category is not None, "no index found for category '{}'".format(path[1])
                    train_labels.append(category)

                    with archive.open(name) as image_file:
                        image = PIL.Image.open(io.BytesIO(image_file.read()))
                        image.load()
                        train_images.append(image)
                    train_file_names.append(path[-1].lower())
    
    return train_images, train_labels, train_file_names, test_images, test_file_names, category_map
    