import random

import torch
import torch.nn as nn

from utils import read_words, create_batches, to_var
from gated_cnn import GatedCNN


seq_len    = 21
embd_size  = 200
n_layers   = 5
kernel     = (5, embd_size)
out_chs    = 64
batch_size = 128


words = read_words('./data/news.en-00001-of-00100', seq_len, kernel[0])
vocab = []
for w in words:
    if w not in vocab:
        vocab.append(w)
vocab_size = len(vocab)
w2i = {'<unk>': 0}
w2i = dict((w, i) for i, w in enumerate(vocab, 1))
print('vocab_size', len(vocab))
data = [w2i[w] for w in words]
data = create_batches(data, batch_size, seq_len)
split_idx = int(len(data) * 0.8)
training_data = data[:split_idx]
test_data = data[split_idx:]


def train(model, data, test_data, optimizer, loss_fn, n_epoch=10):
    print('=========training=========')
    model.train()
    for epoch in range(n_epoch):
        print('----epoch', epoch)
        random.shuffle(data)
        for batch_ct, (X, Y) in enumerate(data):
            X = to_var(torch.LongTensor(X)) # (bs, seq_len)
            Y = to_var(torch.LongTensor(Y)) # (bs,)
            pred = model(X) # (bs, ans_size)
            # _, pred_ids = torch.max(pred, 1)
            loss = loss_fn(pred, Y)
            if batch_ct % 10 == 0:
                print('loss: {:.4f}'.format(loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('current performance at ecpoh', epoch)
        test(model, test_data)

def test(model, data):
    model.eval()
    counter = 0
    correct = 0
    losses = 0.0
    for batch_ct, (X, Y) in enumerate(data):
        X = to_var(torch.LongTensor(X)) # (bs, seq_len)
        Y = to_var(torch.LongTensor(Y)) # (bs,)
        pred = model(X) # (bs, ans_size)
        loss = loss_fn(pred, Y)
        losses += torch.sum(loss).data[0]
        _, pred_ids = torch.max(pred, 1)
        # print('loss: {:.4f}'.format(loss.data[0]))
        correct += torch.sum(pred_ids == Y).data[0]
        counter += X.size(0)

    print('Test Acc: {:.2f} % ({}/{})'.format(100 * correct / counter, correct, counter))
    print('Test Loss: {:.4f}'.format(losses/counter))

model = GatedCNN(seq_len, vocab_size, embd_size, n_layers, kernel, out_chs, vocab_size)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adadelta(model.parameters())
loss_fn = nn.NLLLoss()
train(model, training_data, test_data, optimizer, loss_fn)
# test(model, test_data)
