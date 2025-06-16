"""
Data loading and preprocessing module for IPF (Idiopathic Pulmonary Fibrosis) CT scan analysis.
This module provides classes for handling both single CT scans and longitudinal CT scan sequences.
"""

# Standard library imports
import os
import os.path as osp
import math
import random
import pickle
import warnings
import glob
import argparse

# Third-party imports
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl
from scipy import ndimage
import nibabel as nib
from sklearn.model_selection import KFold
from PIL import Image

class IPFData(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling IPF CT scan data.
    Supports both single CT scans and longitudinal CT scan sequences.
    """
    
    def __init__(self, args, shuffle=True):
        """
        Initialize the IPF data module.
        
        Args:
            args: Configuration arguments
            shuffle (bool): Whether to shuffle the data
        """
        super().__init__()
        self.args = args
        self.shuffle = shuffle

    @property
    def n_classes(self):
        """Get the number of classes in the dataset."""
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        """
        Create appropriate dataset based on configuration.
        
        Args:
            train (bool): Whether to create training or validation dataset
            
        Returns:
            Dataset: Either IPFCTDataset or IPFLongitudinalCTDataset
        """
        # Handle single CT scans
        if hasattr(self.args, 'CT_scans') and self.args.CT_scans:
            if not osp.isdir(self.args.data_root):
                raise NotImplementedError("Data root must be a directory")
                
            # Load and split patient data
            labels = pd.read_csv(self.args.data_root + self.args.label_path)
            patients = labels['patient_id'].unique()
            train_patients, test_patients = self._split_patients(patients)
            
            # Create appropriate dataset
            Dataset = IPFCTDataset
            data_transform = None
            
            if train:
                return Dataset(self.args.data_root, self.args.img_path, 
                             self.args.mask_path, self.args.label_path,
                             self.args, train_patients, data_transform)
            else:
                return Dataset(self.args.data_root, self.args.img_path,
                             self.args.mask_path, self.args.label_path,
                             self.args, test_patients, data_transform)
                             
        # Handle longitudinal CT scans
        elif hasattr(self.args, 'longitudinal_CT_scans') and self.args.longitudinal_CT_scans:
            if not osp.isdir(self.args.data_root):
                raise NotImplementedError("Data root must be a directory")
                
            # Load and split patient data
            labels = pd.read_csv(self.args.data_root + self.args.label_path)
            patients = labels['patient_id'].unique()
            train_patients, test_patients = self._split_patients(patients)
            
            # Create appropriate dataset
            Dataset = IPFLongitudinalCTDataset
            data_transform = None
            
            if train:
                return Dataset(self.args.data_root, self.args.img_path,
                             self.args.mask_path, self.args.label_path,
                             self.args, train_patients, data_transform,
                             self.args.max_longitudinal_CT)
            else:
                return Dataset(self.args.data_root, self.args.img_path,
                             self.args.mask_path, self.args.label_path,
                             self.args, test_patients, data_transform,
                             self.args.max_longitudinal_CT)

    def _split_patients(self, patients):
        """
        Split patients into training and testing sets using K-fold cross-validation.
        
        Args:
            patients: Array of patient IDs
            
        Returns:
            tuple: (train_patients, test_patients)
        """
        kfold_splits = KFold(n_splits=5, shuffle=True, random_state=self.args.random_seed)
        split_idx = kfold_splits.split(np.arange(len(patients)))
        
        # Use first fold for train/test split
        for fold, (train_idx, test_idx) in enumerate(split_idx):
            if fold == 0:
                return patients[train_idx], patients[test_idx]

    def _dataloader(self, train):
        """
        Create a DataLoader for the dataset.
        
        Args:
            train (bool): Whether to create training or validation dataloader
            
        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        dataset = self._dataset(train)
        
        # Handle distributed training
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, 
                num_replicas=dist.get_world_size(), 
                rank=dist.get_rank()
            )
        else:
            sampler = None
            
        return data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=True if train else False
        )

    def train_dataloader(self):
        """Get training dataloader."""
        return self._dataloader(True)

    def val_dataloader(self):
        """Get validation dataloader."""
        return self._dataloader(False)

    def test_dataloader(self):
        """Get test dataloader."""
        return self.val_dataloader()
    
    def predict_dataloader(self):
        """Get prediction dataloader."""
        return self._dataloader(False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Add data-specific arguments to the parser.
        
        Args:
            parent_parser: Parent argument parser
            
        Returns:
            ArgumentParser: Parser with added arguments
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        # Data paths
        parser.add_argument('--data_root', type=str, default="../DnR-master/", help="path to data root")
        parser.add_argument('--img_path', type=str, default="scans_512x512_MyData", help="path to ct data")
        parser.add_argument('--mask_path', type=str, default="LungMasks", help="path to mask")
        parser.add_argument('--label_path', type=str, default="MortalityData.csv", help="path to survival label")
        
        # Data parameters
        parser.add_argument('--sequence_length', type=int, default=16)
        parser.add_argument('--resolution', type=int, default=64)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--max_longitudinal_CT', type=int, default=4)
        parser.add_argument('--random_seed', type=int, default=1234)
        
        # Data type flags
        parser.add_argument('--CT_scans', action='store_true')
        parser.add_argument('--longitudinal_CT_scans', action='store_true')
        parser.add_argument('--mode', type=str, default='interpolation')
        
        return parser

class IPFLongitudinalCTDataset(data.Dataset):
    """
    Dataset class for handling longitudinal CT scans of IPF patients.
    Processes multiple CT scans taken over time for each patient.
    """
    
    def __init__(self, root_dir, img_path, mask_path, label_path, args, patients, spatial_transforms=None, max_longitudinal_CT=2):
        """
        Initialize the longitudinal CT dataset.
        
        Args:
            root_dir (str): Root directory containing the data
            img_path (str): Path to CT scan images
            mask_path (str): Path to lung masks
            label_path (str): Path to patient labels
            args: Configuration arguments
            patients: List of patient IDs to include
            spatial_transforms: Optional spatial transformations to apply
            max_longitudinal_CT (int): Maximum number of CT scans to include per patient
        """
        labels = pd.read_csv(root_dir + label_path)
        self.mode = args.mode

        # Filter labels based on time window
        labels = labels[labels.time_from_baseline < args.time_window_max]
        self.labels = labels
        self.patient_IDs = []
        
        # Filter patients based on mode and available scans
        if self.mode == 'interpolation':
            for patient in patients:
                patient_img_path_list = self.labels[self.labels.patient_id == patient]['image_path'].to_numpy()
                patient_observed_time_points = round(self.labels[self.labels.patient_id == patient]['time_from_baseline']/182.5).to_numpy()
                patient_img_path_list = np.unique(patient_observed_time_points)
                if len(patient_img_path_list) > 2:
                    self.patient_IDs.append(patient)
        elif self.mode == 'extrapolation':
            for patient in patients:
                patient_img_path_list = self.labels[self.labels.patient_id == patient]['image_path'].to_numpy()
                patient_observed_time_points = round(self.labels[self.labels.patient_id == patient]['time_from_baseline']/182.5).to_numpy()
                patient_img_path_list = np.unique(patient_observed_time_points)
                if len(patient_img_path_list) > 2:
                    self.patient_IDs.append(patient)
        elif self.mode == 'reconstruction':
            for patient in patients:
                patient_img_path_list = self.labels[self.labels.patient_id == patient]['image_path'].to_numpy()
                if len(patient_img_path_list) > 1:
                    self.patient_IDs.append(patient)
                    
        # Initialize dataset parameters
        self.dataformat = 'nii'
        self.max_longitudinal_CT = max_longitudinal_CT
        self.root_dir = root_dir
        self.input_D = args.sequence_length
        self.input_before_crop = args.resolution + 30
        self.spatial_transforms = spatial_transforms
        self.outputWidth = args.resolution
        print(f"Processing {len(self.patient_IDs)} patients")

    def axis_nii_to_npy(self, img):
        """
        Convert NIfTI image axis orientation to numpy array format.
        
        Args:
            img: NIfTI image data
            
        Returns:
            numpy.ndarray: Reoriented image data
        """
        img = np.flip(img, axis=1)
        img = np.flip(img, axis=0)
        img = img.transpose(2, 1, 0)
        return img
    
    def __toTensor__(self, data):
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            data: Input numpy array
            
        Returns:
            torch.Tensor: Tensor with shape [1, z, y, x]
        """
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")   
        new_data = torch.from_numpy(new_data) 
        return new_data     

    def __len__(self):
        """Get the number of patients in the dataset."""
        return len(self.patient_IDs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the patient
            
        Returns:
            dict: Dictionary containing:
                - longitudianl_CT_scans: Tensor of CT scans
                - longitudianl_lung_masks: Tensor of lung masks
                - observed_time_points: Array of time points
                - patientID: Patient identifier
                - patient_img_path_list: List of image paths
        """
        # Get patient data
        patient = self.patient_IDs[idx]
        patient_observed_time_points = np.empty((self.max_longitudinal_CT))
        patient_observed_time_points.fill(np.nan)
        
        # Get patient's CT scans and mask
        patient_img_path_list = self.labels[self.labels.patient_id == patient]['image_path'].to_numpy()
        patient_mask_path = self.labels[self.labels.patient_id == patient]['mask_path'].to_numpy()[0]
        patient_observed_time_points_temp = round(self.labels[self.labels.patient_id == patient]['time_from_baseline']/182.5).to_numpy()

        # Get unique time points
        _, indices_of_unique_values = np.unique(patient_observed_time_points_temp, return_index=True)
        patient_img_path_list = patient_img_path_list[indices_of_unique_values]
        patient_observed_time_points_temp = patient_observed_time_points_temp[indices_of_unique_values]

        # Process CT scans
        processed_CTs = torch.zeros(self.max_longitudinal_CT, self.input_D, self.outputWidth, self.outputWidth)
        
        # Handle cases with more scans than max_longitudinal_CT
        if len(patient_img_path_list) > self.max_longitudinal_CT:
            start = np.random.choice(range(len(patient_img_path_list) - self.max_longitudinal_CT), 1, replace=False)
            index = np.arange(start, start + self.max_longitudinal_CT)
            patient_img_path_list = patient_img_path_list[index]
            patient_observed_time_points = patient_observed_time_points_temp[index] - min(patient_observed_time_points_temp[index])
            new_index = np.argsort(patient_observed_time_points)
            patient_observed_time_points = patient_observed_time_points[new_index]
            patient_img_path_list = patient_img_path_list[new_index]
        else:
            patient_observed_time_points[0:len(patient_observed_time_points_temp)] = patient_observed_time_points_temp
            
        # Process each CT scan
        for i in range(len(patient_img_path_list)):
            processed_CT, processed_mask = self.preprosessing_CT(patient_img_path_list[i], patient_mask_path)
            processed_CTs[i,:] = processed_CT
            
        return {
            'longitudianl_CT_scans': processed_CTs,
            'longitudianl_lung_masks': processed_mask,
            'observed_time_points': patient_observed_time_points,
            'patientID': patient,
            'patient_img_path_list': list(patient_img_path_list)
        }

    def preprosessing_CT(self, img_path, mask_path):
        if self.dataformat == 'nii':
            img = nib.load(img_path)
            img = np.array(img.dataobj).astype(np.float32)
            img = self.axis_nii_to_npy(img)
            mask = nib.load(mask_path)
            mask = np.array(mask.dataobj).astype(np.float32)
            mask = self.axis_nii_to_npy(mask)
            top_lung_location = img.shape[0] - 1
            bottom_lung_location = 0
            Dead = torch.tensor(0).to(torch.int64)
            followUpTime = torch.tensor(0).to(torch.float32)
        img, mask = self.__mask_data__(img, mask, top_lung_location, bottom_lung_location)  
        if self.spatial_transforms is not None:          
            if img.shape[0] != self.input_D or img.shape[1] != self.input_before_crop or img.shape[2] != self.input_before_crop:            
                img =  self.__resize_data__(img, [self.input_D, self.input_before_crop, self.input_before_crop])
                mask =  self.__resize_data__(mask, [self.input_D, self.input_before_crop, self.input_before_crop])
        else:
            if img.shape[0] != self.input_D or img.shape[1] != self.outputWidth or img.shape[2] != self.outputWidth:            
                img =  self.__resize_data__(img, [self.input_D, self.outputWidth, self.outputWidth])
                mask =  self.__resize_data__(mask, [self.input_D, self.outputWidth, self.outputWidth])

        img_array = self.__toTensor__(img)
        mask_array = self.__toTensor__(mask)

        if self.spatial_transforms is not None:
            both_images = torch.cat((img_array, mask_array), 0)
            # Apply the transformations to both images simultaneously:
            transformed_images = self.spatial_transforms(both_images)
            # Get the transformed images:
            img_transformed = transformed_images[0].unsqueeze(0)
            mask_transformed = transformed_images[1].unsqueeze(0)
        else:
            img_transformed = img_array
            mask_transformed = mask_array
   
        output_mask = mask_transformed
        img_transformed = img_transformed - 0.5
        return img_transformed, output_mask
             

    def __resize_data__(self, data, targetSize):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [targetSize[0]*1.0/depth, targetSize[1]*1.0/height, targetSize[2]*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data
    
    def __mask_data__(self, data, mask, top_lung_location, bottom_lung_location):
        """
        Resize the data to the input size
        """ 
        # data[mask==0]=0
        while mask[bottom_lung_location,:].sum() < 0:
            bottom_lung_location = bottom_lung_location + 1
        while mask[top_lung_location,:].sum() < 0:
            top_lung_location = top_lung_location - 1

        data = data[bottom_lung_location:top_lung_location,:]
        mask = mask[bottom_lung_location:top_lung_location,:]

        return data, mask
    
class IPFCTDataset(data.Dataset):

    def __init__(self, root_dir, img_path, mask_path, label_path, args, patients, spatial_transforms=None):
        temp = os.listdir(root_dir + img_path)[0]
        if temp.find('.nii') > 0:
            self.dataformat = 'nii'
            labels = pd.read_csv(root_dir + label_path)
            # labels = labels[labels.time_from_baseline<(365.25*3)]
            self.labels = labels
            self.temp_list = self.labels[self.labels.patient_id.isin(patients)]['image_path']
            self.img_list  = [temp_file_path.replace('Leuven_IPF_body_replaced', img_path) for temp_file_path in self.temp_list]             
            self.mask_list = [temp_file_path.replace('Leuven_IPF_body_replaced', mask_path) for temp_file_path in self.temp_list]
            print("Processing {} patients, {} CT scans".format(len(patients), len(self.img_list)))
        else:
            self.dataformat = 'npy'
            self.img_list = [root_dir + img_path + "/" + pth for pth in os.listdir(root_dir + img_path) if len((pth).split('.')) == 2] 
            self.ID_list = [pth.split('_')[0] for pth in os.listdir(root_dir + img_path) if len((pth).split('.')) == 2]
            self.mask_list = [root_dir + mask_path + f"/{patientID}_lungmasks.npy" for patientID in self.ID_list]    
            labels = pd.read_csv(root_dir + label_path)
            labels = labels.set_index('CTCode')
            labels = labels.loc[self.ID_list, ['Dead', 'Follow-up Time', 'top_lung_location', 'bottom_lung_location']]
            self.labels = labels
            self.Dead = labels['Dead'].to_numpy(dtype=np.int64)
            self.followUpTimes = labels['Follow-up Time'].to_numpy(dtype=np.float32)
            self.top_lung_locations = labels['top_lung_location'].to_numpy(dtype=np.int64)
            self.bottom_lung_locations = labels['bottom_lung_location'].to_numpy(dtype=np.int64)
        self.root_dir = root_dir
        self.input_D = args.sequence_length
        self.input_before_crop = args.resolution + 30
        self.spatial_transforms = spatial_transforms
        self.outputWidth = args.resolution
        print("Processing {} datas".format(len(self.img_list)))

    def axis_nii_to_npy(self, img):
        img = np.flip(img, axis=1)
        img = np.flip(img, axis=0)
        img = img.transpose(2, 1, 0)
        return img
    
    def __toTensor__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")   
        new_data = torch.from_numpy(new_data) 
        return new_data     

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.dataformat == 'nii':
            img = nib.load(self.img_list[idx])
            img = np.array(img.dataobj).astype(np.float32)
            img = self.axis_nii_to_npy(img)
            mask = nib.load(self.mask_list[idx])
            mask = np.array(mask.dataobj).astype(np.float32)
            mask = self.axis_nii_to_npy(mask)
            top_lung_location = img.shape[0] - 1
            bottom_lung_location = 0
            Dead = torch.tensor(0).to(torch.int64)
            followUpTime = torch.tensor(0).to(torch.float32)
        else:
            img = np.load(self.img_list[idx])
            mask = np.load(self.mask_list[idx])
            mask = mask > 0
            top_lung_location = self.top_lung_locations[idx]
            bottom_lung_location = self.bottom_lung_locations[idx]
            Dead = torch.tensor(self.Dead[idx]).to(torch.int64)
            followUpTime = torch.tensor(self.followUpTimes[idx]).to(torch.float32)
        img, mask = self.__mask_data__(img, mask, top_lung_location, bottom_lung_location)  
        if self.spatial_transforms is not None:          
            if img.shape[0] != self.input_D or img.shape[1] != self.input_before_crop or img.shape[2] != self.input_before_crop:            
                img =  self.__resize_data__(img, [self.input_D, self.input_before_crop, self.input_before_crop])
                mask =  self.__resize_data__(mask, [self.input_D, self.input_before_crop, self.input_before_crop])
        else:
            if img.shape[0] != self.input_D or img.shape[1] != self.outputWidth or img.shape[2] != self.outputWidth:            
                img =  self.__resize_data__(img, [self.input_D, self.outputWidth, self.outputWidth])
                mask =  self.__resize_data__(mask, [self.input_D, self.outputWidth, self.outputWidth])

        img_array = self.__toTensor__(img)
        mask_array = self.__toTensor__(mask)

        if self.spatial_transforms is not None:
            both_images = torch.cat((img_array, mask_array), 0)
            # Apply the transformations to both images simultaneously:
            transformed_images = self.spatial_transforms(both_images)
            # Get the transformed images:
            img_transformed = transformed_images[0].unsqueeze(0)
            mask_transformed = transformed_images[1].unsqueeze(0)
        else:
            img_transformed = img_array
            mask_transformed = mask_array
        
        output_mask = mask_transformed
        img_transformed = img_transformed - 0.5

        return img_transformed, output_mask, Dead, followUpTime
             

    def __resize_data__(self, data, targetSize):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [targetSize[0]*1.0/depth, targetSize[1]*1.0/height, targetSize[2]*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data
    
    def __mask_data__(self, data, mask, top_lung_location, bottom_lung_location):
        """
        Resize the data to the input size
        """ 
        # data[mask==0]=0
        while mask[bottom_lung_location,:].sum() < 0:
            bottom_lung_location = bottom_lung_location + 1
        while mask[top_lung_location,:].sum() < 0:
            top_lung_location = top_lung_location - 1

        data = data[bottom_lung_location:top_lung_location,:]
        mask = mask[bottom_lung_location:top_lung_location,:]

        return data, mask
