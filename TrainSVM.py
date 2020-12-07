import sklearn
from DefinedModels import LogisticRegression
import torch
import copy
import numpy as np
from tqdm import tqdm
def TrainSVM(Xtrain,ytrain):
    SVM_GRID_PARAMS = [
        {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}
    ]
    class_weight = 'balanced'
    clf = sklearn.svm.SVC(class_weight=class_weight,probability=True,gamma='scale',kernel='linear')
    clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, scoring=None, n_jobs=-1, iid=True,
                                              refit=True, cv=3, verbose=3, pre_dispatch='2*n_jobs',
                                              error_score='raise', return_train_score=True)
    clf.fit(Xtrain, ytrain)
    print(clf.best_params_)
    return clf
def TestSVM(Xtest,clf):

    Prediction=clf.predict(Xtest)
    return Prediction
class MYDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Labelpath):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        Data=torch.FloatTensor(self.Datalist[index])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
def TrainNN(n_features,n_classes,Datadir):
    model=LogisticRegression(n_features=n_features,n_classes=n_classes)

    train_data = MYDataset(Datadir + 'Xtrain.npy',Datadir + 'ytrain.npy')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1024, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-06)
    criterion = torch.nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    Best_loss = 10000.0
    for epoch in range(200):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        model.train()
        model = model.cuda()
        for i, (data, label) in enumerate(tqdm(train_loader)):
            data = data.cuda().float()
            result = model(data)
            label = label.cuda()
            loss = criterion(result, label)
            train_loss += loss.item()
            pred = torch.max(result, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
        if (train_acc / (len(train_data)) >= best_acc) and (train_loss / (len(train_data)) < Best_loss):
            best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, './models/NN.pth')
    return './models/NN.pth'
def TestNN(n_features,n_classes,Datadir,ModelPath):
    model=LogisticRegression(n_features=n_features,n_classes=n_classes)
    model.load_state_dict(torch.load(ModelPath))
    model.eval()
    model = model.cuda()
    test_data = MYDataset(Datadir + 'Xtest.npy', Datadir + 'ytest.npy')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1024, shuffle=False)
    Features = []
    for i, (data, label) in enumerate(tqdm(test_loader)):
        data = data.cuda().float()
        result = model(data)
        pred = torch.max(result, 1)[1]
        for num in range(len(pred)):
            Features.append(np.array(pred[num].cpu().detach().numpy()))
    return Features