The two projects described in the readme file, "Facial-Expression-Recognition" and "Autistic-Children-Classification", are both running in python and pytorch in the windows environment.

1. The first phase of two-phase transfer learning

   Project "Facial-Expression-Recognition" is used to realize the first phase of the two-phase transfer learning. It is refer to "facial expression recognition based on deep convolution neural network"(https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch).

The specific modifications to this project are as follows:

(1) In Mainpro_ CK+. Py, adds an IF statement block (from 187 lines to 196 lines) to save a model's parameters to a file with an extension of .pth. The parameter files are in the root directory of the project, including vgg16.pth,VGG19.pth, MobileNetV1.pth and MobileNetV2.pth.

(2) Add MobileNetV1.py and MobileNetV2.py under the "models" folder, which are used to implement MobileNetV1 and MobileNetV2 respectively.

   Enter "Python mainpro_ck+.py --model * * *" in the command line, "* * *" indicates the model name. Use VGG16, VGG19, MobileNetV1 and MobileNetV2 as model names respectively, and run Mainpro_CK+.Py four times, we get four parameter files. These four parameter files are used as the initial parameters of the models in the second phase of two-phase transfer learning.

2. The second phase of two-phase transfer learning

  The project "Autistic-Children-Classification" is used to realize two experiments, the one-phase transfer learning and the two-phase transfer learning.

(1) Description of each folder of the project: in this project, folder "data" stores the children's face data set obtained from Kaggle, and the pictures have been compressed to 44*44. Folder "models" stores the implementation code of vgg16, vgg19, mobilenetv1 and mobilenetv2 models. Folder "pretrainedmodels" stores the model parameters of vgg16, vgg19, mobilenetv1 and mobilenetv2 trained on ck+ facial expression data set in the first phase. Folder "bestmodels" stores 10 mobilenetv1 models and 10 mobilenetv2 models for multi classifier integration experiments. Folder "rocdatafile" stores the data files used to draw the four models' ROC. 

(2) Program description

A.autism_face_classification.py. It is used in one-phase transfer learning experiment and two-phase transfer learning experiment, and can only run one model at a time. The model is specified by assigning a value to the modelstr string of 51 lines. For example, modelstr= "mobilenetv2", then run the mobilenetv2 model. The program has a code segment that is used to import pre trained model parameters. The code segment is about from line 67 to 80,and the code is as follows:
-------------------------------------------------------
parameterfile='./pretrainedModels/'+modelstr+'.pth'
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
        
model_dict.update(state_dict)
model.load_state_dict(model_dict) 
-----------------------------------------------------------

If these lines of code are commented out, then the program realizes the one-phase transfer learning experiment, on the contrary, it realizes the two-phase transfer learning. 

There is a IF statement (aoubt line 188-189 ) that is used to save the model structure and parameters for multi classifier integration experiments. The code is as follows:
------------------------------------------------------------------------------------------------------------------------
if  max_auc>0.9:           
            torch.save(model,'./bestmodels/tmp/'+modelstr+str(round(max_test_acc,4))+"+"+str(round(max_auc,4))+'.pth')
------------------------------------------------------------------------------------------------------------------------

At the end of the program, a piece of code (about from line 219 to line 238) is used to save the data file used to draw ROC. If you want to save the data file, remove the comments of these lines.The code is as follows:
---------------------------------------------------------
if modelstr=="VGG16":
    datafile = open("./VGG16ROC.txt", "w")
    datafile.write(str(fpr1)+'\n')
    datafile.write(str(tpr1)+'\n')
    datafile.close()
elif modelstr=="VGG19":
    datafile = open("./VGG19ROC.txt", "w")
    datafile.write(str(fpr2)+'\n')
    datafile.write(str(tpr2)+'\n')
    datafile.close()
elif modelstr=="MobileNetV1":
    datafile = open("./MobileNetV1ROC.txt", "w")
    datafile.write(str(fpr3)+'\n')
    datafile.write(str(tpr3)+'\n')
    datafile.close()
elif modelstr=="MobileNetV2":
    datafile = open("./MobileNetV2ROC.txt", "w")
    datafile.write(str(fpr4)+'\n')
    datafile.write(str(tpr4)+'\n')
    datafile.close() 
-------------------------------------------------------------

B.drawROC_V2.py. This program is used to draw the ROC curves of vgg16, vgg19, mobilenetv1 and mobilenetv2 in one-phase transfer learning and the ROC curves of two-phase transfer learning.

C.multi_classifer_v1.py. The program is used to realize the experiment of multi classifier integration, and the experimental results are saved in file "10timerecord.txt". Thise file saves various evaluation metrics of mobilenetv1, mobilenetv2 and integrated classifiers of each time, including ACC, AUC, sensitivity, specificity and other evaluation metrics, as well as 10 times average evaluation metrics of mobilenetv1, mobilenetv2 and the integrated classifier.