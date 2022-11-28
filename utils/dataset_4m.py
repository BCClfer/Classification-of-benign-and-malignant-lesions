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
    def __init__(self,d1_root, d2_root, t2_root, nf_root,data_file,train=True):
        self.d2_root = d2_root
        self.t2_root = t2_root
        self.nf_root = nf_root
        self.d1_root=d1_root

        self.train = train

        data_fd=open(data_file,'r')

        label_ls = []
        d1_ls=[]
        d2_ls=[]
        t2_ls=[]
        nf_ls=[]

        self.use_tensor = True
        name_place = 0


        for line in data_fd:
            line = line.replace('\n', '')
            d1_ls.append(line.split(',')[0])
            d2_ls.append(line.split(',')[1])
            t2_ls.append(line.split(',')[2])
            nf_ls.append(line.split(',')[3])

            label_ls.append(int(line.split(',')[4]))

        
        self.d1_ls=d1_ls
        self.t2_ls = t2_ls
        self.d2_ls = d2_ls
        self.nf_ls=nf_ls
        self.label_ls = label_ls


    def get_mri_data(self, mri_path):
        mri_itk = sitk.ReadImage(mri_path, sitk.sitkFloat32)
        mri_np = sitk.GetArrayFromImage(mri_itk)
        mri_np = (mri_np-np.min(mri_np)) /(np.max(mri_np)-np.min(mri_np))
        mri_tr = torch.from_numpy(mri_np)
        mri_tr.unsqueeze_(0)
        return mri_tr

    
    def norm(self,mri_tr):
        mri_tr=(mri_tr-torch.min(mri_tr))/(torch.max(mri_tr)-torch.min(mri_tr))
        mri_tr.unsqueeze_(0)
        return mri_tr

    def __getitem__(self, index):

        self.d1_name=self.d1_ls[index]
        self.d2_name = self.d2_ls[index]
        self.t2_name = self.t2_ls[index]
        self.nf_name=self.nf_ls[index]
        
        d1_path=self.d1_root+self.d1_name
        t2_path = self.t2_root + self.t2_name
        d2_path = self.d2_root + self.d2_name
        nf_path=self.nf_root+self.nf_name

        d2_tr = torch.load(d2_path)
        d2_tr = self.norm(d2_tr) 
        

        nf_tr = torch.load(nf_path)
        nf_tr = self.norm(nf_tr) 
        
        t2_tr = torch.load(t2_path)
        t2_tr = self.norm(t2_tr)

        d1_tr=torch.load(d1_path)
        d1_tr=self.norm(d1_tr)
        
        volume_tr=torch.zeros((4,64,64,64),dtype=torch.float32)
        volume_tr[0]=d1_tr
        volume_tr[1]=d2_tr
        volume_tr[2]=t2_tr
        volume_tr[3]=nf_tr
        
        if self.train:
            return volume_tr, self.label_ls[index]
        else:
            return volume_tr, self.label_ls[index], [self.d1_name,self.d2_name, self.t2_name,self.nf_name]

    def get_name(self, index):
        return self.name

    def __len__(self):
        # print("len(self.mhaFileList)", len(self.mhaFileList))
        return len(self.d2_ls)
