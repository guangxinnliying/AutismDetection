# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#import matplotlib.pyplot as plt
from util.KZDataset import KZDataset
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv1 import MobileNetV1
from models.vgg import VGG
#from models.resnet import ResNet18


# 定义一些超参数
batch_size = 64 
learning_rate = 0.005
num_epoches = 60

norm_mean=[0.485,0.456,0.406]
norm_std=[0.229,0.224,0.225]
transfm=transforms.Compose([transforms.Resize((44,44)),transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
#transfm=transforms.Compose([transforms.Resize((44,44)),transforms.ToTensor()])
samplefile="./data/samplelist_train.txt"  #用于训练及验证
samplefile_test="./data/samplelist_test.txt" #用于评估
total_fold=10  #折数
val_batchs=16
test_batchs=16
total_max_val_acc=0.0
total_max_test_acc=0.0
total_max_sensitivity=0.0
total_max_specificity=0.0
total_max_G_Mean=0.0
total_max_error_rate=0.0
total_max_F_Measure=0.0
total_max_auc=0.0
total_max_fall_out=0.0
max_folds_auc=0.0

if torch.cuda.is_available():
    use_cuda=True
    print("cuda is available!!!")

#测试集不分折，所以写在循环之外.
testset=KZDataset(txt_path=samplefile_test,ki=0,K=total_fold,typ='test',transform=transfm,rand=False)
test_loader=DataLoader(dataset=testset,batch_size=test_batchs,shuffle=True,drop_last=False)  
break_flag=False
modelstr="MobileNetV2"
#选择预训练模型
if modelstr=="VGG19":
    parameterfile='./pretrainedModels/VGG19.pth'
elif modelstr=="VGG16":   
    parameterfile='./pretrainedModels/VGG16.pth'
elif modelstr=="Resnet18":    
    parameterfile='./pretrainedModels/Resnet18.pth'
elif modelstr=="MobileNetV1":    
    parameterfile='./pretrainedModels/MobileNetV1.pth'    
elif modelstr=="MobileNetV2":    
    parameterfile='./pretrainedModels/MobileNetV2.pth' 

time1 = time.time() #记录模型开始训练时间
#foldfilename='./bestmodels/'+modelstr+'10FoldIndexRecord.txt' #记录10个折的各种指标
#foldrecordfile=open(foldfilename,'w')
for current_fold in range(total_fold):
    print("Current fold is:",current_fold)
    #装载数据    
    #print("total_fold=",total_fold)
    trainset=KZDataset(txt_path=samplefile,ki=current_fold,K=total_fold,typ='train',transform=transfm,rand=False)
    valset=KZDataset(txt_path=samplefile,ki=current_fold,K=total_fold,typ='val',transform=transfm,rand=False)
    
    train_loader=DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,drop_last=False)
    val_loader=DataLoader(dataset=valset,batch_size=val_batchs,shuffle=True,drop_last=False)

    #len(train_loader)的长度是所有训练数据样本数量除以每批数量（batch_size)，同样，
    #len(val_loader)的长度是所有验证数据样本数量除以每批数量。
    #print("train_loader",len(train_loader),"  len(valset)=",len(valset))
    #sys.exit(0)
    # 选择模型
    if modelstr=="VGG16" or modelstr=="VGG19":
        model = VGG(modelstr)  #调用VGG的__init__()
    elif modelstr=="MobileNetV1" :
            model = MobileNetV1()
    elif modelstr=="MobileNetV2" :
            model = MobileNetV2()
    if use_cuda:
        model = model.cuda()
        
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  #定义交叉熵损失函数,  这个方法集成了softmax计算
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
       
    #导入预训练的模型参数
    print("load pretrained model's paratmeters......")
    pretrained_dict=torch.load(parameterfile)
    model_dict = model.state_dict()
    state_dict={}
    i=0
    j=0

    for k,v in pretrained_dict.items():
        i=i+1
        if (k in model_dict.keys()) and (k[0:k.index('.')]not in('classifier','linear','fc')):
            state_dict[k]=v
            j=j+1
        
    #print("Total number of parameters is: ",i," matching number is: ",j)
  
    #sys.exit()
    
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)    
    
    max_val_acc=0.0000
    max_epoch=0
    max_test_acc=0.0000 
    max_val_acc_test=0.0000
    max_test_acc_val=0.0000
    
    max_sensitivity=0.0
    max_specificity=0.0
    max_G_Mean=0.0
    max_error_rate=0.0
    max_F_Measure=0.0
    max_auc=0.0

    #训练模型   
    print("Begin to train model...")
    for epoch in range(num_epoches):
        #print("current_fold:",current_fold,"Current epoch:",epoch)
        for i,(inputs,labels) in enumerate(train_loader):
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)
            #print(inputs)
            #sys.exit(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            #if epoch%10 == 0:
                #print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
        
            
        #验证模型
        model.eval()
        val_loss = 0.0000
        val_acc = 0.0000
        num_correct = 0
        for data in val_loader:
            img, label = data
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            loss = criterion(out, label)
            #print("out:",out) 
            #print("label:",label) 
            val_loss += loss.data.item()*label.size(0)
            _, pred = torch.max(out, 1)
            #print("pred=",pred,"pred.shape=",pred.shape,"pred.shape[0]=",pred.shape[0])
            num_correct = (pred == label).sum()
            val_acc += num_correct.item()
            #print("num_correc=",num_correct," eval_acc=",eval_acc)
            
    
        #在测试集上评估模型
        model.eval()
        test_loss = 0.0000
        test_acc = 0.0000
        num_correct = 0
        row_i=0
        y=[9]*len(testset) #len(testset) 是测试集中包含图片的数量
        pred_per=[9.0000]*len(testset)    
        TP=0
        TN=0
        FP=0
        FN=0
        for data in test_loader:
            img, label = data
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            loss = criterion(out, label)
            test_loss += loss.data.item()*label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            test_acc += num_correct.item()
            
            #数组y是真实值，和标签数组是一样的。只是标签在一次循环中是test_batchs个数，要把它们放到y中
            percentage=torch.nn.functional.softmax(out,dim=1)
            if percentage[0,0]==float("nan"):
                print("percentage:",percentage)
            
            if len(label)==test_batchs: #前面的批都够一个test_batchs，例如总数104，一个batchs是20，那么最后一个不够20个
                endval=test_batchs
            else:
                endval=len(testset)-row_i*test_batchs
            
            for i in range(endval):  
                y[row_i*test_batchs+i]=label[i].item()
                pred_per[row_i*test_batchs+i]=round(percentage[i,pred[i].item()].item(),4)
                if (label[i].item()==1 and pred[i].item()==1):
                    TP=TP+1
                if (label[i].item()==0 and pred[i].item()==0):
                    TN=TN+1
                if (label[i].item()==1 and pred[i].item()==0):
                    FN=FN+1
                if (label[i].item()==0 and pred[i].item()==1):
                    FP=FP+1                  
        
            row_i=row_i+1
        
        #每评估一次，就计算一次sensitivity等指标
        sensitivity=TP/(TP+FN) #即TPR，可以通过求acc同时获得
        specificity=TN/(TN+FP) 
        G_Mean=(sensitivity*specificity)**0.5  #开平方根
        error_rate=(FP+FN)/(TP+TN+FP+FN)  #即FPR，可以通过求acc同时获得
        F_Measure=2*TP/(2*TP+FP+FN)
        fpr,tpr,thresholds=roc_curve(y,pred_per,pos_label=1)
        roc_auc = round(auc(fpr,tpr),4)
    
        fall_out=FP/(TN+FP)   #fall_out=1-specificity
        val_acc=val_acc /len(valset)  #当前epoch的验证准确率
        test_acc=test_acc /len(testset) #当前epoch的测试准确率
        
        if val_acc>max_val_acc:
            max_val_acc=val_acc       #当前验证准确率取代最大验证准确率    
            max_val_epoch=epoch
            max_val_acc_test=test_acc #取得最大验证准确率时的测试准确率

        if test_acc>max_test_acc: 
            max_test_acc=test_acc  #当前测试准确率取代最大测试准确率
            max_test_acc_val=val_acc  #取得最大测试准确率时的验证准确率
            max_val_epoch=epoch
            max_epoch=epoch
            max_sensitivity=sensitivity
            max_specificity=specificity
            max_error_rate=error_rate
            max_auc=roc_auc            
            max_G_Mean=G_Mean
            max_F_Measure=F_Measure
            #下面这句话用来保存模型            
            torch.save(model,'./bestmodels/'+modelstr+'F'+str(current_fold)+'.pth')            
            if max_auc>max_folds_auc:
                    max_folds_auc=max_auc
                    roc_fpr=fpr
                    roc_tpr=tpr
    '''         
    #下面的if语句表示如果记录了每个折的比较好的模型后就直接退出该折
    if break_flag:
        break;
   '''
    #输出每个折的最好验证准确率和最好测试准确率(已经跳出60个epoch循环)        
    total_max_val_acc=total_max_val_acc+max_val_acc #累加每个折的最好测试准确率
    total_max_test_acc+=max_test_acc #累加每折最大测试准确率
    total_max_sensitivity+=max_sensitivity
    total_max_specificity+=max_specificity
    total_max_G_Mean+=max_G_Mean
    total_max_error_rate+=max_error_rate
    total_max_F_Measure+=max_F_Measure
    total_max_auc+=max_auc
    
    print("max_val_acc:",max_val_acc)
    print("max_test_acc:",max_test_acc)
    print("max_auc:",max_auc)
    print("max_error_rate:",max_error_rate)
    print("max_sensitivity:",max_sensitivity)
    print("max_specificity:",max_specificity)
    print("max_F_Measure:",max_F_Measure)
    print("max_G_Mean:",max_G_Mean)
    
    '''
    foldrecordfile.write('current_fold:'+str(current_fold)+'\n')
    foldrecordfile.write('max_test_acc:'+str(round(max_test_acc,4))+'\n')
    foldrecordfile.write('max_auc:'+str(round(max_auc,4))+'\n')
    foldrecordfile.write('max_error_rate:'+str(round(max_error_rate,4))+'\n')
    foldrecordfile.write('max_sensitivity:'+str(round(max_sensitivity,4))+'\n')
    foldrecordfile.write('max_specificity:'+str(round(max_specificity,4))+'\n')
    foldrecordfile.write('max_F_Measure:'+str(round(max_F_Measure,4))+'\n')
    foldrecordfile.write('max_G_Mean:'+str(round(max_G_Mean,4))+'\n')
    '''

avg_G_Mean=((total_max_sensitivity/total_fold)*(total_max_specificity/total_fold))**0.5
print("Avg total_best_val_acc of 10 folds is:",total_max_val_acc/total_fold)  #输出10个折的平均最好验证准确率  
print("Avg total_best_test_acc of 10 folds is:",total_max_test_acc/total_fold)  #输出10个折的平均最好测试准确率  
print("Avg total_max_sensitivity of 10 folds is:",total_max_sensitivity/total_fold)  #输出10个折的平均最好sensitivity  
print("Avg total_max_specificity of 10 folds is:",total_max_specificity/total_fold)  #输出10个折的平均最好specificity  
print("Avg total_max_error_rate of 10 folds is:",total_max_error_rate/total_fold)  #输出10个折的平均最好error_rate 
print("Avg total_max_auc of 10 folds is:",total_max_auc/total_fold)  #输出10个折的平均最好auc  
print("Avg total_max_G_Mean of 10 folds is:",avg_G_Mean)  #输出10个折的平均最好G_Mean  
print("Avg total_max_F_Measure of 10 folds is:",total_max_F_Measure/total_fold)  #输出10个折的平均最好F_Measure  
print("Avg total_max_fall_out of 10 folds is:",total_max_fall_out/total_fold)  #输出10个折的平均最好fall_out  

time2 = time.time()
#输出整数，秒数 
print("Time spent is:"+str(int(time2-time1)))

#foldrecordfile.close()

'''
rocdatafile='./rocdatafile/multiclass/'+modelstr+'_transfer4FC(lr0.01).txt'
f=open(rocdatafile,'w')
f.write(str(fpr)+'\n')
f.write(str(tpr)+'\n')
f.close()
'''



