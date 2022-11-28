import os
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
from torch.utils import data
from torch.utils.data import DataLoader
import natsort
from scipy import stats
import random
import natsort

np.set_printoptions(threshold=1000000)
torch.set_printoptions(threshold=10000000000, sci_mode=False, precision=5)


class dataset(data.Dataset):
    def __init__(self, root, data_file,train=True):
        self.root = root
        data_file = open(data_file, 'r')

        mri_ls = []
        label_ls = []

        self.use_tensor = False
        self.train = train


        for line in data_file:
            line = line.replace('\n', '')
            mri_ls.append(line.split(',')[0])
            label_ls.append(int(line.split(',')[1]))


        if mri_ls[0].endswith('.pt'):
            self.use_tensor = True

        self.mri_ls = mri_ls
        self.label_ls = label_ls


        # for key in self.ml_feature.keys():
        #     print(key)

    def get_mri_data(self, mri_path):
        mri_itk = sitk.ReadImage(mri_path, sitk.sitkFloat32)
        mri_np = sitk.GetArrayFromImage(mri_itk)
        mri_np=mri_np/np.max(mri_np)
        mri_tr = torch.from_numpy(mri_np)
        mri_tr.unsqueeze_(0)
        return mri_tr

  

    def __getitem__(self, index):

        self.name = self.mri_ls[index]
        mri_path = self.root + self.name
        #print(mri_path)
        if self.use_tensor:
            mri_tr = torch.load(mri_path)
            mri_tr=(mri_tr-torch.min(mri_tr))/(torch.max(mri_tr)-torch.min(mri_tr))
            mri_tr.unsqueeze_(0)

        else:
            mri_tr = self.get_mri_data(mri_path)


        # print("fe", fe)
        # fe=(fe-torch.min(fe))/(torch.max(fe)-torch.min(fe))
        # print("fe",fe)

        if self.train:
            return mri_tr, self.label_ls[index]
        else:
            return mri_tr,self.label_ls[index], self.name,

    def get_name(self, index):
        return self.name

    def __len__(self):
        # print("len(self.mhaFileList)", len(self.mhaFileList))
        return len(self.mri_ls)
