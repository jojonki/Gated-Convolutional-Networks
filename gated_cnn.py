import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 seq_len,
                 vocab_size,
                 embd_size,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 ans_size):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        # self.embd_size = embd_size

        self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv_0      = nn.Conv2d(1, out_chs, kernel, padding=(2, 0)) # )2, 99
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0)) # )2, 99

        # todo bias

        self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs*seq_len, ans_size)

    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        x = self.embedding(x) # (bs, word_len, embd_size)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x) # (bs, Cout, seq_len, 1?)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1?)
        h = A * F.sigmoid(B) # (bs, Cout, seq_len, 1?)
        res_input = h # TODO this is h1 not h0

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h)
            B = conv_gate(h)
            h = A * F.sigmoid(B) # (bs, Cout, seq_len, 1?)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, -1) # (bs, Cout*seq_len)
        out = self.fc(h) # (bs, ans_size)
        out = F.log_softmax(out)

        return out
