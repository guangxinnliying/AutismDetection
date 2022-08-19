The two projects described in the readme file, that are "Facial-Expression-Recognition" and "Autistic-Children-Classification", are both running in the windows environment.

1. The first phase of two-phase transfer learning
   Project "Facial-Expression-Recognition" is used to realize the first phase of the two-phase transfer learning. It is refer to "facial expression recognition based on deep convolution neural network"(https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch).
The specific modifications to this project are as follows:
(1) In Mainpro_ CK+. Py, the last line "torch.save(net.state_dict(), './pretrainedmodels/'+opt.model+'.pth')" is used to save a model's parameters to a file with an extension of .pth. The parameter files are in the folder "pretrainedmodels" of the project, including MobileNetV2.pth and MobileNetV3_Large.pth.

(2) Add MobileNetV2.py and MobileNetV3_Large.py under the "models" folder, which are used to implement MobileNetV2 and MobileNetV3_Large respectively.
   Enter "Python mainpro_ck+.py --model * * *" in the command line, "* * *" indicates the model name. Use MobileNetV2 and MobileNetV3_Large as model names respectively, and run Mainpro_CK+.Py four times, we get their parameter files. The two parameter files are used as the initial parameters of the models in the second phase of the two-phase transfer learning.

2. The second phase of the two-phase transfer learning
  The project "Autistic-Children-Classification" is used to realize two experiments, that are the one-phase transfer learning and the two-phase transfer learning.
(1) Description of each folder of the project: in this project, folder "data" stores the children's face data set obtained from the kaggle platform. Folder "models" stores the implementation code of mobilenetv2 and mobilenetv3_Large models. Folder "pretrainedmodels" stores the model parameters of mobilenetv2 and mobilenetv3_Large trained on CK+ facial expression data set in the first phase. Folder "bestmodels" stores mobilenetv2 and mobilenetv3_Large for multi-classifier integration experiments. 

(2) Program description
A.autism_face_classificatio.py. It is used in the one-phase transfer learning experiment and the two-phase transfer learning experiment. The program can only run one model at a time. The model is specified by assigning a value to the modelstr string of 52 lines, that is "modelstr="MobileNetV3_Large". For example, modelstr= "MobileNetV3_Large", then run the MobileNetV3_Large model. The program has a code segment (from 74 lines to 92 lines) that is used to import pre-trained model parameters. If these lines of code are commented out, then the program realizes the one-pase transfer learning experiment, on the contrary, it realizes the two-phase transfer learning. Code line 200-201 is used to save the model structure and parameters for multi-classifier integration experiments.

B.draw_line_char.py. This program is used to draw the broken lines of accuracy and AUC,  the two metrics is about 10 times of training of 
mobilenetv2 and mobilenetv3_Large in the one-phase transfer learning and the two-phase transfer learning.

C.multi_classifer_v1.py. The program is used to realize the experiment of multi-classifier integration, and the experimental results are saved in file "resultrecord.txt". This file saves various evaluation metrics of mobilenetv2, mobilenetv3_Large and integrated classifiers, including accuracy, AUC, sensitivity, specificity and other evaluation metrics. To run multi_classifer_v1.py, run autism_face_classification.py first to find a mobilenetv2 and a mobilenetv3_Large with fine performance, and put the two model file in folder "bestmodels". 