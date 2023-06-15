import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import requests
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
look_back = 50

response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt') 
raw_text = response.text.split('\n\n\n\n')
raw_text = raw_text[1:]
raw_text = " ".join(raw_text)
raw_text = raw_text.replace('\n',' ')
raw_text = raw_text.replace('  ',' ')
import re
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
#set the size of the dataset
length = look_back
x = []
y = np.array([])
counter = 0
for t in tokens:
    print(counter)
    d = np.array(t)
    
    for i in range(len(d)-length):
        x.append(d[i:i+length])
    y = np.append(y,d[length:])
    counter +=1
    if counter == 1000:
        break
y = y.reshape((len(y),1))
x = np.array(x)
#create unique numerical token for each unique word
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)
seq_length = x.shape[1]

X = np.reshape(x, (x.shape[0], seq_length, 1))
X = torch.tensor(X, dtype=torch.float32)
X = X / float(vocab_size)
y = torch.tensor(y)




class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        x = self.relu(x)
        # produce output
        x = self.linear(self.dropout(x))
        # x = self.softmax(x)
        return x

n_epochs = 100
batch_size = 256
learning_rate = 20
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=batch_size)

# Split train test
generator1 = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(loader, [0.7,0.3],generator1)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_set:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model,reverse_word_map], f"{str(batch_size)}-{str(learning_rate)}-single-char.pth")

