import numpy
from tifffile import tifffile
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
import cv2

#find a dedicate device if available
device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "mps" if torch.mps.is_available() else "cpu"
print("using device: ", device)


def loadThermalImages():
    noFirePath = Path("data/No Fire/Thermal/Celsius TIFF/")
    firePath = Path("data/Fire/Thermal/Celsius TIFF/")

    noFireFiles = [noFirePath / f.name for f in noFirePath.iterdir() if f.is_file()]
    fireFiles = [firePath / f.name for f in firePath.iterdir() if f.is_file()]

    # load the no fire images:
    noFireImages = tifffile.imread(noFireFiles)
    noFireClass = np.zeros(len(noFireFiles))
    # load the fire images:
    fireImages = tifffile.imread(fireFiles)
    fireClass = np.ones(len(fireFiles))


    indices = np.linspace(0, len(fireFiles), len(noFireFiles), endpoint=False, dtype=int)
    mixed = np.insert(fireImages, indices, noFireImages, axis=0)
    mixedClass = np.insert(fireClass, indices, noFireClass)

    return mixed, mixedClass

def loadRGBImages():
    noFirePath = Path("data/No Fire/RGB/Corrected FOV/")
    firePath = Path("data/Fire/RGB/Corrected FOV/")

    noFireFiles = [noFirePath / f.name for f in noFirePath.iterdir() if f.is_file()]
    fireFiles = [firePath / f.name for f in firePath.iterdir() if f.is_file()]

    # load the no fire images:
    noFireImages = np.array([cv2.imread(file) for file in noFireFiles], dtype=np.float32)/256
    noFireClass = np.zeros(len(noFireFiles))
    # load the fire images:
    fireImages = np.array([cv2.imread(file) for file in fireFiles], dtype=np.float32)/256
    fireClass = np.ones(len(fireFiles))


    indices = np.linspace(0, len(fireFiles), len(noFireFiles), endpoint=False, dtype=int)
    mixed = np.insert(fireImages, indices, noFireImages, axis=0)
    mixedClass = np.insert(fireClass, indices, noFireClass)

    return mixed, mixedClass

def trainValTestSplit(data :np.ndarray, labels: np.ndarray):
    rng = np.random.default_rng(seed=67)
    lbls = [1,2,3]
    split_freq = [0.7,0.1,0.2]
    split_lbls = rng.choice(lbls, size=data.shape[0], p=split_freq)
    train_mask = [n == 1 for n in split_lbls]
    val_mask = [n == 2 for n in split_lbls]
    test_mask = [n == 3 for n in split_lbls]

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    train_labels = labels[train_mask]
    val_labels = labels[val_mask]
    test_labels = labels[test_mask]
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

class GenericThermalDataset(Dataset):
    def __init__(self, images, labels):
        super(GenericThermalDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]).unsqueeze(0).to(device), torch.from_numpy(np.array([self.labels[index]], dtype=np.float32)).to(device)

class GenericRGBDataset(Dataset):
    def __init__(self, images, labels):
        super(GenericRGBDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #todo reshuffle channel order
        return torch.from_numpy(self.images[index]).to(device), torch.from_numpy(np.array([self.labels[index]], dtype=np.float32)).to(device)

def creatThermalDataLoaders():
    #step 1 load the data
    imgs, labels = loadThermalImages()
    #step 2 create teh splits
    train_data, train_labels, val_data, val_labels, test_data, test_labels = trainValTestSplit(imgs, labels)

    #step 3 create the datasets
    train_dataset = GenericThermalDataset(train_data, train_labels)
    val_dataset = GenericThermalDataset(val_data, val_labels)
    test_dataset = GenericThermalDataset(test_data, test_labels)

    batch_size = 16

    #step4 create the data loaders
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def createRGBDataLoaders():
    #step 1 load the data
    imgs, labels = loadRGBImages()
    #step 2 create teh splits
    train_data, train_labels, val_data, val_labels, test_data, test_labels = trainValTestSplit(imgs, labels)

    #step 3 create the datasets
    train_dataset = GenericRGBDataset(train_data, train_labels)
    val_dataset = GenericRGBDataset(val_data, val_labels)
    test_dataset = GenericRGBDataset(test_data, test_labels)

    batch_size = 16

    #step4 create the data loaders
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

thermal_train_data_loader:DataLoader
thermal_validation_data_loader:DataLoader
thermal_test_data_loader:DataLoader

rgb_train_data_loader:DataLoader
rgb_validation_data_loader:DataLoader
rgb_test_data_loader:DataLoader

"""Load the thermal images as ready to use data loaders
"""
def loadThermalData():
    global thermal_train_data_loader, thermal_validation_data_loader, thermal_test_data_loader
    thermal_train_data_loader, thermal_validation_data_loader, thermal_test_data_loader = creatThermalDataLoaders()

"""Load the rgb images as ready to use data loaders
"""
def loadRGBData():
    global rgb_train_data_loader, rgb_validation_data_loader, rgb_test_data_loader
    rgb_train_data_loader, rgb_validation_data_loader, rgb_test_data_loader = createRGBDataLoaders()
