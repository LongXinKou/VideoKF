import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

class Detector_Head(nn.Module):
    """docstring for Detector_Head."""

    def __init__(self, input_channel=512,max_length=200,mode='detector'):
        super(Detector_Head, self).__init__()
        self.max_length = max_length
        self.rnn = nn.LSTM(input_channel,128,bidirectional=True,batch_first=True)
        if mode == 'detector':
            self.linear_1 = nn.Linear(256,1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear_1 = nn.Linear(max_length * 256,2)
        self.mode = mode
    def forward(self,x,lengths):
        x = pack_padded_sequence(x,lengths=lengths,batch_first=True, enforce_sorted=True)
        h0 = torch.zeros(2, x.batch_sizes[0], 128) # 2 for bidirection
        c0 = torch.zeros(2, x.batch_sizes[0], 128)
        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        x,_ = self.rnn(x,(h0,c0))
        x,_ = pad_packed_sequence(x,padding_value=0,total_length=self.max_length, batch_first=True)
        if self.mode == 'detector':
            b,f,c = x.size()
            x = x.contiguous().view(b*f,c)
            x = self.linear_1(x)
            x = self.sigmoid(x)
            x = x.view(b,f)
            x = pack_padded_sequence(x,lengths=lengths,batch_first=True, enforce_sorted=True)
            x,_ = pad_packed_sequence(x,padding_value=0,total_length=lengths[0], batch_first=True)
            return x
        else:
            b,f,c = x.size()
            x = x.contiguous().view(b,f*c)
            x = self.linear_1(x)
            return x


class Detector(nn.Module):
    """docstring for Detector."""

    def __init__(self, input_channel=512, max_length=200):
        super(Detector, self).__init__()
        self.head = Detector_Head(input_channel=input_channel, max_length=max_length, mode='detector')
    def forward(self,x,length):
        return self.head(x,length)


class Predictor(nn.Module):
    """docstring for Trajectory_Sequence_Detector."""

    def __init__(self, input_channel=512, max_length=200):
        super(Predictor, self).__init__()
        self.head = Detector_Head(input_channel=input_channel, max_length=max_length, mode='evaluator')
    def forward(self,x,length):
        return self.head(x,length)
