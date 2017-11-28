import torch
from torch.autograd import Variable


def read_words(fpath, seq_len, filter_h):
    words = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            # TODO only choose specified length sentence
            if len(tokens) == seq_len - 2:
                words.extend((['<pad>']*int(filter_h/2)) + ['<s>'] + tokens + ['</s>'])

    return words


def create_batches(data, batch_size, seq_len):
    ret_data = []
    X, Y = [], []
    for i in range(0, len(data)-(seq_len+1), seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    for i in range(0, len(X)-batch_size, batch_size):
        ret_data.append((X[i:i+batch_size], Y[i:i+batch_size]))
    return ret_data


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
