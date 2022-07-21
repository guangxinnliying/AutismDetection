# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data.dataset import *
from PIL import Image
from torch.nn import functional as F
import random

class KZDataset(Dataset):
    def __init__(self,txt_path=None,ki=0,K=10,typ='train',transform=None,rand=False):
        '''
        特别注意：__init__ 是正确的，_nit_ 是错误的，运行时报错
        txt_path:数据文件，这里形式为（单张图片路径及文件名 类别）
            img1.png  0
            ng  1
            ...
        ki:当前是第几折，从0开始，范围[0,k)
        K:总的折数
        typ:用于区分训练集和验证集
        transform:对图片的数据增强
        rand:是否随机
        '''
        self.all_data_info=self.get_img_info(txt_path)
        #print("self.all_data_info:",self.all_data_info)
        if rand:
            random.seed(1)
            random.shuffle(self.all_data_info)
        leng=len(self.all_data_info)
        every_z_len=leng//K
        #b1=0
        #b2=0
        if typ=='test':
            self.data_info=self.all_data_info
        elif typ=='val':
            self.data_info=self.all_data_info[every_z_len*ki:every_z_len*(ki+1)-1]
            #b1=every_z_len*ki
            #b2=every_z_len*(ki+1)-1
        elif typ=='train':
            if ki==K-1:
                self.data_info=self.all_data_info[:every_z_len*ki-1]
                #b1=0
                #b2=every_z_len*ki-1
            else:
                self.data_info=self.all_data_info[:every_z_len*ki]+self.all_data_info[every_z_len*(ki+1):]
                '''
                b0=0
                b1=every_z_len*ki
                b2=every_z_len*(ki+1)
                b3=2285
                '''
        self.transform=transform
        #print("typ=",typ," self.data_info.len=",len(self.data_info),"b1=",b1,"  b2=",b2)

    def __getitem__(self,index):
		#Dataset读取图片的函数
        img_pth,label=self.data_info[index]
        img=Image.open(img_pth).convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        return img,label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(txt_path):
		#解析输入的txt函数
		#转为二维list存储，每一维维[图片路径，图片类别]
        data_info=[]
        data=open(txt_path,'r')
        data_lines=data.readlines()
        for data_line in data_lines:
            data_line=data_line.split()
            img_path=data_line[0]
            label=int(data_line[1])
            data_info.append((img_path,label))
        return data_info
	

