import torch
import numpy as np
from PIL import Image
import pandas as pd
import os



class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, csv_file, image_directory):

        #image_directory = "../cassava-leaf-disease-classification/train_images"
        #csv_file = "../"
        
        self.csv_file = pd.read_csv(csv_file)
        self.image_directory = image_directory

    def __getitem__(self, index):
        filename = self.csv_file.iloc[index][0]
        image_path = self.image_directory + '/' + filename
        im = Image.open(image_path, 'r')
        im = im.resize((224, 224))
        im = np.array(im)
        im = torch.from_numpy(im)

        im = im / 255

        im = torch.transpose(im, 0, 2)
        
        label = self.csv_file.iloc[index][1]

        # print(im.size())

        return im, label

    def __len__(self):
        return self.csv_file.shape[0]


# print("Hello world")
# train_dataset = StartingDataset('../cassava-leaf-disease-classification/train.csv', '../cassava-leaf-disease-classification/train_images')
# inp, lab = train_dataset[261]
# print(inp)
# print(lab)