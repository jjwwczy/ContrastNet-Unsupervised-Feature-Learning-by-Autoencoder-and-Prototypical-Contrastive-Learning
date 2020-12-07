# 1. 数据预处理，包括读数据，生成数组文件等等
#
import argparse
from Preprocess import Preprocess,PerClassSplit,splitTrainTestSet
from TrainAE import TrainAAE_patch,SaveFeatures_AAE,TrainVAE_patch,SaveFeatures_VAE
from TrainPCL import TrainContrast, ContrastPredict
import numpy as np
import joblib
from TrainSVM import TrainSVM,TestSVM, TrainNN, TestNN
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import time
import csv
from utils import reports
dataset_names = ['IP','SA','PU']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='SA', choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--train',type=int, default=0,choices=(0,1))
parser.add_argument('--perclass', type=float, default=5) #会除以100
parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0","cuda:1"))
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--encoded_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--temperature', type=float, default=1)#程序中会除以100
parser.add_argument('--Windowsize', type=int, default=27)
parser.add_argument('--classifier', type=str, default='linear',choices=("linear","svm"))
parser.add_argument('--train_contrast',type=int, default=0,choices=(0,1))
parser.add_argument('--Patch_channel', type=int, default=15)
parser.add_argument('--RandomSeed', type=bool, default=False)#True 代表固定随机数， False代表不固定


args = parser.parse_args()

if args.RandomSeed:
    randomState=345
else:
    randomState=int(np.random.randint(1, high=1000))
args.temperature=args.temperature/100
args.perclass=args.perclass/100
print(args)
output_units = 9 if (args.dataset == 'PU' or args.dataset == 'PC') else 16
Datadir='./DataArray/'
XPath = Datadir + 'X.npy'
yPath = Datadir + 'y.npy'
# # 2. 生成两个自编码器并训练，保存好模型参数
train_start = time.time()
if args.train:

    Preprocess(XPath, yPath, args.dataset, args.Windowsize, Patch_channel=args.Patch_channel)
    TrainVAE_patch(XPath, Patch_channel=args.Patch_channel, windowSize=args.Windowsize, encoded_dim=args.encoded_dim,
                  batch_size=args.batch_size)
    TrainAAE_patch(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)

# # 3. 用已保存的自编码器将两种数据编码成两个特征数组存起来
test_start1 = time.time()
VAEPath=Datadir+'VAE_Features.npy'
AAEPath=Datadir+'AAE_Features.npy'
VAEFeatures=SaveFeatures_VAE(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
np.save(VAEPath,VAEFeatures)
AAEFeatures=SaveFeatures_AAE(XPath,Patch_channel=args.Patch_channel,windowSize=args.Windowsize,encoded_dim=args.encoded_dim,batch_size=args.batch_size)
np.save(AAEPath,AAEFeatures)
test_stop1 = time.time()
#############训练对比网络
if args.train_contrast==True:
    TrainContrast(AAEPath,VAEPath,batch_size=128,temperature=args.temperature,projection_dim=args.projection_dim,input_dim=args.encoded_dim)
train_stop = time.time()
test_start2 = time.time()
ContrastFeature=ContrastPredict(AAEPath,VAEPath,batch_size=128,temperature=args.temperature,projection_dim=args.projection_dim,input_dim=args.encoded_dim)
test_stop2 = time.time()
np.save(Datadir+'ContrastFeature.npy',ContrastFeature)
y = np.load(yPath)
stratify = np.arange(0, output_units, 1)


for feature in ["Contrast","AAE","VAE"]:
    if feature == "Contrast":
        fea = ContrastFeature
        projection_dim=args.projection_dim
    elif feature == "AAE":
        fea = AAEFeatures
        projection_dim=1024
    elif feature == "VAE":
        fea = VAEFeatures
        projection_dim = 1024
    if args.perclass > 1:
        Xtrain, Xtest, ytrain, ytest = PerClassSplit(fea, y, args.perclass, stratify,randomState=randomState)
    else:
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(fea, y, 1 - args.perclass,randomState=randomState)
    np.save(Datadir + 'Xtrain.npy', Xtrain)
    np.save(Datadir + 'ytrain.npy', ytrain)
    np.save(Datadir + 'Xtest.npy', Xtest)
    np.save(Datadir + 'ytest.npy', ytest)
    if args.classifier=='svm':
        train_start2 = time.time()
        SVM_model=TrainSVM(Xtrain,ytrain)
        train_stop2 = time.time()
        joblib.dump(SVM_model, './models/SVM.model')
        SVM_model=joblib.load('./models/SVM.model')
        test_start3 = time.time()
        Predictions=TestSVM(Xtest,SVM_model)
        test_stop3 = time.time()
    else:
        train_start2 = time.time()
        ModelPath=TrainNN(n_features=projection_dim,n_classes=output_units,Datadir=Datadir)
        train_stop2 = time.time()

        test_start3 = time.time()
        Predictions=TestNN(n_features=projection_dim,n_classes=output_units, ModelPath=ModelPath,Datadir=Datadir)
        test_stop3 = time.time()
#####算上训练分类器和测试分类器的时间
    # TrainTime=(train_stop-train_start)+(train_stop2-train_start2)
    # TestTime=(test_stop1-test_start1)+(test_stop2-test_start2)+(test_stop3-test_start3)
######不计分类器的训练和测试时间损耗
    TrainTime = (train_stop - train_start)
    TestTime=(test_stop1-test_start1)+(test_stop2-test_start2)
    ytest=np.load(Datadir+'ytest.npy')
    classification = classification_report(ytest.astype(int), Predictions)
    print(classification)
    classification, confusion, oa, each_acc, aa, kappa = reports(Predictions, ytest.astype(int), args.dataset)
    ## 9. 存储分类报告至csv，以及画出效果图
    # file_name = "AECLR_IP"+'_'+str(perclass)+"perclass.txt"
    # with open(file_name, 'w') as x_file:
    #     x_file.write('{}'.format(classification))
    each_acc_str = ','.join(str(x) for x in each_acc)
    add_info=[args.dataset,args.perclass,args.temperature,args.Windowsize,args.classifier,feature,oa,aa,kappa,TrainTime,TestTime]+each_acc_str.split('[')[0].split(']')[0].split(',')
    csvFile = open("compare.csv", "a")
    writer = csv.writer(csvFile)
    writer.writerow(add_info)
    csvFile.close()
