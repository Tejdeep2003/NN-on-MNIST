
import torch.nn as nn 
import torch.optim as optim 
import torchmetrics
import torch
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection 
import torch.utils.data as data 
import numpy as np 
from tqdm import tqdm
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
submission = np.load("sample_submission.npy")
#@PROTECTED_1

X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test = preprocessing.MinMaxScaler().fit_transform(X_test)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def randomizeData(X, y):
  np.random.seed(9)
  separate = np.random.rand(X.shape[0]) < np.percentile(np.random.rand(X.shape[0]), 90)
  Ytrain = y[separate]
  Xtrain = X[separate]
  Xtest =  X[~separate]
  Ytest = y[~separate]  
  return Xtrain, Ytrain, Xtest, Ytest

trainx, trainy, testx, testy = randomizeData(X_train, y_train)



X_test = torch.from_numpy(X_test).float().cuda()
trainx = torch.from_numpy(trainx).float().cuda()
trainy = torch.from_numpy(trainy).float().cuda()
testx = torch.from_numpy(testx).float().cuda()
testy = torch.from_numpy(testy).float().cuda()


tdata = []
for i in range(len(trainx)):
   tdata.append([trainx[i],trainy[i]])

model = nn.Sequential(
    nn.Linear(3072,1024),
    
    nn.ELU(alpha=1.5),
    nn.Dropout(0.25),

    nn.Linear(1024,512),
    
    nn.ELU(alpha=1.5),
    nn.Dropout(0.25),

    nn.Linear(512,256),
    
    nn.ELU(alpha=1.5),
    nn.Dropout(0.25),

    nn.Linear(256,128),
    
    nn.ELU(alpha=1.5),
    nn.Dropout(0.25),

    nn.Linear(128,64),
    
    nn.ELU(alpha=1.5),
    nn.Dropout(0.25),
    
    nn.Linear(64, 10),
)

model = model.to('cuda')
optimizer = optim.Adam(model.parameters(),lr=0.001)
lfn = nn.CrossEntropyLoss().cuda()
epochs = 250
bs=500

loss_values = []
train_dataloader = data.DataLoader(tdata, bs, shuffle=True)
for epoch in tqdm(range(epochs)):
    model.train()
    for X, y in train_dataloader:
        predictions = model(X)
        loss = lfn(predictions,y.long())
        optimizer.zero_grad()
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

model.eval()

train_res = model(trainx)
train_res = torch.argmax(train_res, axis=1)
print(train_res.type)
acc = torchmetrics.Accuracy(task="multiclass",num_classes=10).to('cuda')
print(acc(trainy, train_res))


test_res = model(testx)
test_res = torch.argmax(test_res, axis=1)
acc = torchmetrics.Accuracy(task="multiclass",num_classes=10).to('cuda')
print(acc(testy, test_res))

testing_res = model(X_test)
testing_res = torch.argmax(testing_res, axis=1)
testing_res = testing_res.cpu().detach().numpy()
submission = testing_res

