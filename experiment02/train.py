import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import requests
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import re
import string
from time import time
import argparse
def getResDir():
    import os
    from datetime import datetime
    curr_work_dir = os.path.dirname(__file__)
    resDir = os.path.join(curr_work_dir,'results')
    # datetime.now().strftime("%b%d_%H-%M-%S")
    # print(resDir)
    return resDir
log_dir = getResDir()
comment = "New Day"
writer = SummaryWriter(
    log_dir=log_dir,
    comment=comment,
    )






parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--bz', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ts', type=int, default=10, help='time step')
parser.add_argument('--hs', type=int, default=8, help='Hidden neurons')
parser.add_argument('--nl', type=int, default=2, help='num Layers')
parser.add_argument('--m', type=str, default='LSTM', help='model')
args = parser.parse_args()

start = time()
response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt') 
raw_text = response.text.split('\n\n\n\n')
# response = open("C:/Users/Ribuzari/Desktop/shakespeare.txt",'r')
# raw_text = response.read().split('\n\n\n\n')
raw_text = raw_text[1]
raw_text = "".join(raw_text)
raw_text = raw_text.replace('\n',' ')
raw_text = raw_text.replace('  ',' ')
pattern_order = r'[0-9]'
raw_text = re.sub(pattern_order, '', raw_text)
raw_text = re.sub(',', ' ,', raw_text)
raw_text = re.sub(r'[(]', '( ', raw_text)
raw_text = re.sub(r'[)]', ' )', raw_text)
raw_text = re.sub(':', ' :', raw_text)
raw_text = re.sub(r'[?]', ' ?', raw_text)
raw_text = re.sub(r'[.]', ' .', raw_text)
raw_text = re.sub(r'[!]', ' !', raw_text)
raw_text = raw_text.split('            ')[1:]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_text)
tokens = tokenizer.texts_to_sequences(raw_text)
raw_text = None
#set the size of the dataset
n_epochs = 2000
batch_size = args.bz
learning_rate = args.lr
hidden_size = args.hs
num_layers = args.nl
look_back = args.ts
length = look_back
name = args.m + ' total run'
x = []
y = np.array([])
counter = 0
for t in tokens:
    d = np.array(t)
    
    for i in range(len(d)-length):
        x.append(d[i:i+length])
    y = np.append(y,d[length:])
    counter +=1
    if counter <= -1:
        break
print(counter)
y = y.reshape((len(y),1))
x = np.array(x)
#create unique numerical token for each unique word
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)
seq_length = x.shape[1]

x = np.reshape(x, (x.shape[0], seq_length, 1))
# x = torch.tensor(x, dtype=torch.float32)
x = x / float(vocab_size)
# y = torch.tensor(y)



class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        if args.m == 'LSTM':
            self.layer = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        elif args.m == 'RNN':
            self.layer = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, _ = self.layer(x)
        # take only the last output
        x = x[:, -1, :]
        # x = self.relu(x)
        # produce output
        x = self.linear(self.dropout(x))
        # x = self.softmax(x)
        return x

class SonarDataset(Dataset):
    def __init__(self, x, y):
        # convert into PyTorch tensors and remember them
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.x[idx]
        target = self.y[idx]
        return features, target
    

print('lb' + str(look_back)+'+h'+str(hidden_size)+'+n'+str(num_layers) + '+lr' + str(learning_rate))
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
dataset = SonarDataset(x, y)

# # Split train test
# generator1 = torch.Generator().manual_seed(42)
# trainset, testset = torch.utils.data.random_split(dataset, [0.7,0.3],generator1)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0)
# train_size = len(train_loader.dataset)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0)
# test_size = len(test_loader.dataset)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
train_size = len(train_loader.dataset)


time_ppdt = time() - start
loss_array = []
tloss_array = []
start = time()
best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    tloss = 0
    # with torch.no_grad():
    #     for tX_batch, ty_batch in train_loader:
    #         ty_pred = model(tX_batch.to(device))
    #         tloss += loss_fn(ty_pred, ty_batch.to(device))
    #     tloss /= train_size
    #     print("train: %.8f" % (tloss))
    #     tloss_array.append(tloss.cpu().item())
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch.to(device))
            loss += loss_fn(y_pred, y_batch.to(device))
        loss /= train_loader
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: test: %.8f" % (epoch, loss),end=' ')
        loss_array.append(loss.cpu().item())
    writer.add_scalars( 
            main_tag=name,
            tag_scalar_dict = {
                'lb' + str(look_back)+'+h'+str(hidden_size)+'+n'+str(num_layers) + '+lr' + str(learning_rate): tloss.cpu().item(),
                },
            global_step=epoch
            )
    # with torch.no_grad():
    #     for tX_batch, ty_batch in train_loader:
    #         ty_pred = model(tX_batch.to(device))
    #         tloss += loss_fn(ty_pred, ty_batch.to(device))
    #     tloss /= train_size
    #     print("train: %.8f" % (tloss))
    #     tloss_array.append(tloss.cpu().item())
    # writer.add_scalars( 
    #         main_tag=name,
    #         tag_scalar_dict = {
    #             'test': loss.cpu().item(),
    #             'train': tloss.cpu().item(),
    #             },
    #         global_step=epoch
    #         )
time_train = time()-start
# Tensor.cpu() 
# import pandas as pd
# df = pd.DataFrame(data={'loss': loss_array,'train loss': tloss_array })
# df.to_csv(str(batch_size) + '+' + str(learning_rate) +'RNN-loss'+'.csv',  index=None)

torch.save([best_model,reverse_word_map], args.m + '+' + 'lb' + str(look_back)+'+h'+str(hidden_size)+'+n'+str(num_layers) + '+lr' + str(learning_rate) + ".pth")
# f = open(str(batch_size) + '+' + str(learning_rate) + "RNN-timeRecord.txt", "w")
# f.write(f'''Model Summary:
#         {model}
#         {'-'*50}

#         Epoch {epoch}: Cross-entropy: {loss} 
#         Training preparation time: {time_ppdt} seconds
#         Training time: {time_train} seconds''') 
# f.close()
