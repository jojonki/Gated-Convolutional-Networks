import os
import torch
from torch.autograd import Variable


def read_words(data_dir, seq_len, filter_h):
    words = []
    for file in os.listdir(data_dir):
        print('Load', file)
        with open(os.path.join(data_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split()
                # TODO currently only choose specified length sentence
                if len(tokens) == seq_len - 2:
                    # TODO i'm not sure about the padding...
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
