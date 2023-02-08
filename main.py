import os
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
random.seed(114514)
os.environ['PYTHONHASHSEED'] = str(114514)
torch.manual_seed(114514)
torch.cuda.manual_seed(114514)
torch.cuda.manual_seed_all(114514)

data_path = r'data.xlsx'
data = pd.read_excel(data_path, header=0).values
X_train = data[20:145, 0:5]
Y_train = data[20:145, 5]
X_validation = data[145:165, 0:5]
Y_validation = data[145:165, 5]
X_test = data[0:20, 0:5]
Y_test = data[0:20, 5]
X_train = torch.tensor(X_train).to(device=device)
Y_train = torch.tensor(Y_train).to(device=device).reshape(-1, 1)
X_train = X_train.to(torch.float)
Y_train = Y_train.to(torch.float)
X_validation = torch.tensor(X_validation).to(device=device)
Y_validation = torch.tensor(Y_validation).to(device=device).reshape(-1, 1)
X_validation = X_validation.to(torch.float)
Y_validation = Y_validation.to(torch.float)
X_test = torch.tensor(X_test).to(device=device)
Y_test = torch.tensor(Y_test).to(device=device).reshape(-1, 1)
X_test = X_test.to(torch.float)
Y_test = Y_test.to(torch.float)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(5, 14),
            nn.LeakyReLU(),
            nn.Linear(14, 14),
            nn.LeakyReLU(),
            nn.Linear(14, 14),
            nn.Sigmoid(),
            nn.Linear(14, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.stack(x)
        return y


model = NN().to(device)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 100000

lossf = list()
losst = list()
mini = 10
for i in range(epochs):
    model.train()
    Y_predict = model(X_train)
    loss = loss_function(Y_predict, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        lossf.append(loss)
        model.eval()
        Y_validationpredict = model(X_validation)
        losst.append(loss_function(Y_validationpredict, Y_validationpredict))
        if mini > loss_function(Y_validationpredict, Y_validationpredict):
            mini = loss_function(Y_validationpredict, Y_validationpredict)
            torch.save(model.state_dict(), "model.pth")

lossf = torch.tensor(lossf).cpu()
plt.plot(range(lossf.size(0)), lossf)
losst = torch.tensor(losst).cpu()
plt.plot(range(losst.size(0)), losst)
plt.show()

model = NN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
Y_testpredict = model(X_validation)
print(Y_testpredict)
err = Y_validation.mul(1-Y_testpredict) + (1-Y_validation).mul(Y_testpredict)
print(err.sum(0)/err.size(0))
Y_validation = torch.tensor(Y_validation).cpu()
Y_testpredict = torch.tensor(Y_testpredict).cpu()
plt.scatter(range(err.size(0)), Y_validation)
plt.scatter(range(err.size(0)), Y_testpredict)
plt.show()

Y_testpredict = model(X_test)
print(Y_testpredict)
err = Y_test.mul(1-Y_testpredict) + (1-Y_test).mul(Y_testpredict)
print(err.sum(0)/err.size(0))
Y_test = torch.tensor(Y_test).cpu()
Y_testpredict = torch.tensor(Y_testpredict).cpu()
plt.scatter(range(err.size(0)), Y_test)
plt.scatter(range(err.size(0)), Y_testpredict)
plt.show()
