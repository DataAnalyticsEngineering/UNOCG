"""
Data loading for microstructure datasets
"""

import math
from typing import Optional, Union, Dict

import torch
import h5py


class MicrostructureDataset(torch.utils.data.Dataset):
    """
    Represents a dataset in a microstructure from a HDF5 file
    """
    def __init__(
        self,
        file_name: str,
        group_name: str,
        lazy_loading: Optional[bool] = True,
        device = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Constructor of the class. Create a `PyTorch` dataset from given HDF5 file groups.

        :param file_name: path to the HDF5 file
        :type file_name: str
        :param group_name: path to the group in the HDF5 file
        :type group_name: str
        :param lazy_loading:
        :type lazy_loading: bool
        :param dtype:
        """
        super().__init__()
        self.file_name = file_name
        self.group_name = group_name
        self.lazy_loading = lazy_loading
        self.keys = []
        self.loaded_keys = []
        self.images = {}
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        self.device = device
        self.tensor_args = {"dtype": self.dtype, "device": self.device}

        with h5py.File(self.file_name, "r") as file:
            for dset_name in file[self.group_name].keys():
                self.keys.append(dset_name)
        if not self.lazy_loading:
            for dset_name in self.keys:
                self.load_dset(dset_name)

    def __len__(self) -> int:
        """
        Get the length of the dataset, i.e. how many data points it contains.

        :return: Length of the dataset
        :rtype: int
        """
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.tensor, str]]:
        """
        Fetch a data point with given index from the dataset

        :param index: Index of the data point
        :type index: int
        :return: microstructure image
        :rtype: torch.Tensor
        """
        if index >= len(self.keys):
            raise ValueError("Dataset is not available")
        dset_name = self.keys[index]
        self.load_dset(dset_name, force_loading=False)
        return self.images[dset_name].clone()

    def load_dset(self, dset_name: str, force_loading: Optional[bool] = True):
        """
        Load dataset from HDF5 file

        :param dset_name:
        :param force_loading:
        :return:
        """
        if (not force_loading) and (dset_name in self.loaded_keys):
            return
        
        with h5py.File(self.file_name, "r") as file:
            image = torch.tensor(file[self.group_name][dset_name]["image"][...], **self.tensor_args)
        
        self.images[dset_name] = image
        self.loaded_keys.append(dset_name)
