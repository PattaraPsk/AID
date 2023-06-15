import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import requests
import re
from torchtext.data import get_tokenizer
import torchdata.datapipes as dp
import torchtext.transforms as T

from torchtext.vocab import build_vocab_from_iterator
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
model_file = 'C:/Users/Ribuzari/Documents/myProj/myGit/AID/256-0.001-single-char.pth'
best_model, reverse_word_map  = torch.load(model_file)

response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt') 
raw_text = response.text.split('\n\n\n\n')
response = None
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
raw_text = raw_text.split('             ')[1:]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_text)
tokens = tokenizer.texts_to_sequences(raw_text)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
print()
vocab_size = len(tokenizer.word_index) + 1
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.RNN(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
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
    
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(best_model)

my_token = tokens[150]
seed_text= my_token[0:50]
# seed_text= my_token



for word in seed_text:
     result = reverse_word_map.get(word)
     print(result, end=" ")
with torch.no_grad():
    for i in range(50):
        # format input array of int into PyTorch tensor
        px = np.reshape(seed_text, (1, len(seed_text), 1)) / float(vocab_size)
        px = torch.tensor(px, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(px.to(device))
        # convert logits into one character
        index = int(prediction.argmax())
        result = reverse_word_map.get(index)
        print(result, end=" ")
        # append the new character into the prompt for the next iteration
        seed_text = np.append(seed_text,index)
        # seed_text = seed_text[1:]
print()
print("Done.")