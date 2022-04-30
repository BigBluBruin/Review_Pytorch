from pathlib import Path
from pyexpat import model
import requests
import pickle
import gzip
import torch
import math
from IPython.core.debugger import set_trace
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
import numpy as np

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")



x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 1  # how many epochs to train for
n, c = x_train.shape
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# hand writing:
# def log_softmax(x):
#     return x - x.exp().sum(-1).log().unsqueeze(-1)
# def nll(input, target):
#     return -input[range(target.shape[0]), target].mean()
# loss_func = nll
# equivalent in torch.nn.functional (F)
loss_func = F.cross_entropy





def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def get_model():
    model = Minist_Logistic()
    return model, optim.SGD(model.parameters(),lr=lr)


# weights = torch.randn(784, 10) / math.sqrt(784)
# weights.requires_grad_()
# bias = torch.zeros(10, requires_grad=True)
# handwritten version:
# def model(xb):
#     return log_softmax(xb @ weights + bias)

# subclass nn.Module
class Minist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        # --use nn.Parameters to define parameters--
        # self.weights = nn.Parameter(torch.randn(784,10)/math.sqrt(784))
        # self.bias = nn.Parameter(torch.zeros(10))
        # -- use nn embedded function to define parameters--
        self.lin = nn.Linear(784,10)


    def forward(self, xb):
        # -- nn.parameter version ---
        # return xb@self.weights+self.bias
        # -- nn.forward version ---
        return self.lin(xb)


#model = Minist_Logistic() #initialized object
model, opt = get_model()




xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
yb = y_train[0:bs]
print(preds[0], preds.shape)
print(loss_func(preds, yb))
print(accuracy(preds, yb))


 

train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
valid_ds = TensorDataset(x_valid,y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # for i in range((n - 1) // bs + 1):
        #     #set_trace()
        #     start_i = i * bs
        #     end_i = start_i + bs
        #     # xb = x_train[start_i:end_i]
        #     # yb = y_train[start_i:end_i]
        #     xb, yb = train_ds[i*bs:i*bs+bs]
        model.train() #set the train mode
        for xb,yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
            # with torch.no_grad():
            #     # --hand written version --
            #     # weights -= weights.grad * lr
            #     # bias -= bias.grad * lr
            #     # weights.grad.zero_()
            #     # bias.grad.zero_()
            #     # --use Module class version --
            #     for p in model.parameters():
            #         p-=p.grad*lr
            #     model.zero_grad()
        model.eval() # set the evaluation mode
        # with torch.no_grad():
        #     valid_loss = sum(loss_func(model(xb),yb) for xb, yb in valid_dl)
        losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])



train_dl,valid_dl = get_data(train_ds,valid_dl,bs)
model, opt = get_model()
fit(epochs,model,loss_func,opt,train_dl,valid_dl)

