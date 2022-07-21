# -*- coding: utf-8 -*-
# 该程序用来绘制一阶段迁移和两阶段中四个模型的roc曲线。要画一阶段迁移，就屏蔽掉两步迁移的数据文件；反之亦然。
#The program is used to draw ROC curves of four models in one-phase transfer learning and two-phase transfer learing. 
#To draw the ROC of one-phase transfer learning, annotate the data files of the two-phase transfer learing; vice versa.

import os
import sys
import matplotlib.pyplot as plt

def totalOfDataInFile(file_name,datatype):
    f = open(file_name,'r')               # 返回一个文件对象   
    line = f.readline() 
    total=0
    endflag=False
    charcount=0
    if datatype=='fpr': #读fpr,在文件中是第一个个中括号内的数。
        while line:   
            if line[0]=='[':
                line=line.strip('[')

            if line[-2]==']':
                line=line.strip('\n')  #每行最后一个字符是回车换行
                line=line.strip(']')   #最后一行倒数第二个字符是']'
                endflag=True
    
            datatmp=[float(i) for i in line.split()] 
            total+=len(datatmp)
            if endflag==True:
                break
            line = f.readline()
    elif datatype=='tpr':  #读tpr,在文件中是第二个中括号内的数。
        #print('datatype:',datatype)
        while line:   
            if line[0]=='[':
                charcount+=1
            if charcount==2:
                if line[0]=='[':
                    line=line.strip('[')
                if line[-2]==']':
                    line=line.strip('\n')  #每行最后一个字符是回车换行
                    line=line.strip(']')   #最后一行倒数第二个字符是']'
                    endflag=True
                datatmp=[float(i) for i in line.split()] 
                total+=len(datatmp)
            if endflag==True:
                break
            line = f.readline()
    f.close() 
    return total
    
def getDataFromFile(file_name,datatype):
    total=totalOfDataInFile(file_name,datatype)
    #print('total',total)
    data_list=[0.0]*total
    f = open(file_name,'r')  # 返回一个文件对象    
    line = f.readline() 
    currentlen=0
    charcount=0
    endflag=False
    if datatype=='fpr':#读fpr,在文件中是第一个个中括号内的数。
        while line:  
            if line[0]=='[':
                line=line.strip('[')
            if line[-2]==']':
                line=line.strip('\n')  #每行最后一个字符是回车换行
                line=line.strip(']')   #最后一行倒是第二个字符是']'
                endflag=True
        
            datatmp=[float(i) for i in line.split()] 

            for i in range(len(datatmp)):
                data_list[currentlen+i]=datatmp[i]
        
            currentlen+=len(datatmp)
            if endflag==True:
                break
            line = f.readline()
    elif datatype=='tpr':  #读tpr,在文件中是第二个中括号内的数。
        while line:   
            if line[0]=='[':
                charcount+=1
            if charcount==2:
                if line[0]=='[':
                    line=line.strip('[')
                if line[-2]==']':
                    line=line.strip('\n')  #每行最后一个字符是回车换行
                    line=line.strip(']')   #最后一行倒是第二个字符是']'
                    endflag=True
                datatmp=[float(i) for i in line.split()] 
                for i in range(len(datatmp)):
                    data_list[currentlen+i]=datatmp[i]        
                currentlen+=len(datatmp)
                
            if endflag==True:
                break
            line = f.readline()   
    f.close()
    return data_list

filename="./rocdatafile/VGG16ROC_notransfer(lr0.01).txt"  #指定VGG16一阶段迁移学习的ROC数据文件（Specify the ROC data file of VGG16 in one-phase transfer learning）
#filename="./rocdatafile/VGG16ROC_transfer(lr0.01).txt"  #指定VGG16两阶段迁移学习的ROC数据文件(Specify the ROC data file of VGG16 in two-phase transfer learning)
vgg16_fpr=getDataFromFile(filename,'fpr')
vgg16_tpr=getDataFromFile(filename,'tpr')


filename="./rocdatafile/VGG19ROC_notransfer(lr0.01).txt" #指定VGG19一阶段迁移学习的ROC数据文件（Specify the ROC data file of VGG19 in one-phase transfer learning）
#filename="./rocdatafile/VGG19ROC_transfer(lr0.01).txt" #指定VGG19两阶段迁移学习的ROC数据文件(Specify the ROC data file of VGG19 in two-phase transfer learning)
vgg19_fpr=getDataFromFile(filename,'fpr')
vgg19_tpr=getDataFromFile(filename,'tpr')

filename="./rocdatafile/MobileNetV1ROC_notransfer(lr0.01).txt" #指定MobileNetV1一阶段迁移学习的ROC数据文件(Specify the ROC data file of MobileNetV1 in one-phase transfer learning)
#filename="./rocdatafile/MobileNetV1ROC_transfer(lr0.01).txt" #指定MobileNetV1两阶段迁移学习的ROC数据文件(Specify the ROC data file of MobileNetV1 in two-phase transfer learning)
MoblieNetV1_fpr=getDataFromFile(filename,'fpr')
MoblieNetV1_tpr=getDataFromFile(filename,'tpr')

filename="./rocdatafile/MobileNetV2ROC_notransfer(lr0.005).txt" #指定MobileNetV2一阶段迁移学习的ROC数据文件(Specify the ROC data file of MobileNetV2 in one-phase transfer learning)
#filename="./rocdatafile/MobileNetV2ROC_transfer(lr0.005).txt" #指定MobileNetV2两阶段迁移学习的ROC数据文件(Specify the ROC data file of MobileNetV2 in two-phase transfer learning)
MoblieNetV2_fpr=getDataFromFile(filename,'fpr')
MoblieNetV2_tpr=getDataFromFile(filename,'tpr')


plt.plot(vgg16_fpr, vgg16_tpr, color='green', label='VGG16')
plt.plot(vgg19_fpr, vgg19_tpr, color='red', label='VGG19')
plt.plot(MoblieNetV1_fpr, MoblieNetV1_tpr, color='blue', label='MobileNetV1')
plt.plot(MoblieNetV2_fpr, MoblieNetV2_tpr, color='black', label='MobileNetV2')

plt.legend() # 显示图例
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')