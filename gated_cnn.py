import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self, seq_len, vocab_size, embd_size, kernel, out_chs, ans_size):
        super(GatedCNN, self).__init__()
        self.embd_size = embd_size
        self.embedding = nn.Embedding(vocab_size, embd_size)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv      = nn.Conv2d(1, out_chs, kernel, padding=(2, 0)) # )2, 99
        self.conv_gate = nn.Conv2d(1, out_chs, kernel, padding=(2, 0)) # )2, 99
        # todo bias

        self.conv2      = nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) # )2, 99
        self.conv_gate2 = nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) # )2, 99

        self.fc = nn.Linear(out_chs*seq_len, ans_size)

    def forward(self, x):
        # x: (N, seq_len)
        # Embedding
        bs = x.size(0) # batch size
        # seq_len = x.size(1) # number of words in a sentence
        x = self.embedding(x) # (bs, word_len, embd_size)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin, Hin, Win )
        #    Output: (bs, Cout,Hout,Wout)
        A = self.conv(x) # (bs, Cout, seq_len, 1?)
        B = self.conv_gate(x) # (bs, Cout, seq_len, 1?)
        h0 = A * F.sigmoid(B) # (bs, Cout, seq_len, 1?)

        A2 = self.conv2(h0)
        B2 = self.conv_gate2(h0)
        h1 = A2 * F.sigmoid(B2) # (bs, Cout, seq_len, 1?)

        # todo residual

        hL = h1 # (bs, Cout, seq_len, 1?)
        hL = hL.view(bs, -1) # (bs, Cout*seq_len)
        out = self.fc(hL) # (bs, ans_size)

        out = F.log_softmax(out)
        return out
