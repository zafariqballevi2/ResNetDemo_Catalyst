import torchio
from pathlib import Path
from torchio.transforms import RescaleIntensity, RandomAffine, Compose
import torchvision.transforms as transformss
import torch.nn as nn
import pandas as pd

class data_preprocssing:
   def __init__(self, path_to_csv, path_to_nifti):
       self.data=path_to_csv
       self.dataset_dir=path_to_nifti  
       
   def build(self):
        SEED=42        
        data=pd.read_csv(self.data)
        ab=data.label
        
        ############################################
        transforms = [
            RescaleIntensity((0, 1)),     
            RandomAffine(),
            transformss.ToTensor(),
        ]
        transform = Compose(transforms)
        #############################################
        
        
        
        dataset_dir=self.dataset_dir
        dataset_dir=Path(dataset_dir)
        
        
        
        
        images_dir = dataset_dir
        labels_dir = dataset_dir
        image_paths = sorted(images_dir.glob('**/*.nii'))
        label_paths = sorted(labels_dir.glob('**/*.nii'))
        assert len(image_paths) == len(label_paths)
        
        # These two names are arbitrary
        MRI = 'features'
        BRAIN = 'targets'
        
        #split dataset into training and validation
        from catalyst.utils import split_dataframe_train_test
        
        train_image_paths, valid_image_paths = split_dataframe_train_test(
            image_paths, test_size=0.2, random_state=SEED)
        
        
        #training data
        subjects = []
        i=0
        for (image_path, label_path) in zip(train_image_paths, label_paths):
            subject_dict = {
                MRI: torchio.Image(image_path, torchio.INTENSITY),
                BRAIN: ab[i],
            }
            i=i+1
            subject = torchio.Subject(subject_dict)
            subjects.append(subject)
        train_data = torchio.ImagesDataset(subjects)
        
        
        #validation data
        subjects = []
        for (image_path, label_path) in zip(valid_image_paths, label_paths):
            subject_dict = {
                MRI: torchio.Image(image_path, torchio.INTENSITY),
                BRAIN: ab[i],
            }
            i=i+1
            subject = torchio.Subject(subject_dict)
            subjects.append(subject)
        test_data = torchio.ImagesDataset(subjects)
        return train_data, test_data
        