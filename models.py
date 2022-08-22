import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList
import math

import tqdm

'''LSTM'''
class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """
    def __init__(self, input_size, hidden_size, layer_num, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bias = bias

        linears = [nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)]
        for i in range(self.layer_num-1):
            linears.append(nn.Linear(2 * hidden_size, 4 * hidden_size, bias=bias))
        self.linears = nn.ModuleList(linears)
        
        ingateNorm = [nn.LayerNorm(hidden_size) for i in range(len(self.linears))]
        forgeteNorm = [nn.LayerNorm(hidden_size) for i in range(len(self.linears))]
        cellgateNorm = [nn.LayerNorm(hidden_size) for i in range(len(self.linears))]
        outgateNorm = [nn.LayerNorm(hidden_size) for i in range(len(self.linears))]

        self.ingateNorm = nn.ModuleList(ingateNorm)
        self.forgeteNorm = nn.ModuleList(forgeteNorm)
        self.cellgateNorm = nn.ModuleList(cellgateNorm)
        self.outgateNorm = nn.ModuleList(outgateNorm)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hn, cn):  # hn, cn: (self.layer_num, N, self.hidden_dim),  x: (N, D)
        houtputs, coutputs = [], []
        for i, layer in enumerate(self.linears):
            hx, cx = hn[i, :, :], cn[i, :, :]
            x = torch.cat([hx, x], dim=1)
            gates = layer(x)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate, forgetgate, cellgate, outgate = self.ingateNorm[i](ingate), self.forgeteNorm[i](forgetgate),\
                                                    self.cellgateNorm[i](cellgate), self.outgateNorm[i](outgate)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate2 = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            c = cx * forgetgate + ingate * cellgate2
            x = outgate * torch.tanh(c) + cellgate
            coutputs.append(c.unsqueeze(0))
            houtputs.append(x.unsqueeze(0))
        houtputs = torch.cat(houtputs, dim=0)
        coutputs = torch.cat(coutputs, dim=0)
        return houtputs, coutputs


'''
featureType: Qn features and Ql features and Ql features dim and label_dim
modelType: input_dim, hidden_dim, layer_dim, output_dim

Ql features:
masts: [0., 1., 2., 3.]
educd: [0., 1., 2., 3., 4., 5., 6.]
trdtp:[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
       26., 27., 28., 29.]
poscd: [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 99.]
gender_code: [0., 1.]
age: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
labels: [0, 2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]
'''
class LSTMModel(nn.Module):
    def __init__(self, featureType, modelType, device):
        super(LSTMModel, self).__init__()
        self.device = device
        topN = modelType['topN']
        weight = modelType['weight']
        bias = modelType['bias']
        # features
        self.Qn_num = featureType['Qn']
        self.Qn_dim = featureType['Qn_dim']
        self.Ql = featureType['Ql']
        self.Ql_dim = featureType['Ql_dim']

        self.Ql_num = len(self.Ql)
        self.Ql_class = sum(self.Ql)
        # Hidden dimensions
        self.input_dim = self.Qn_num * self.Qn_dim + self.Ql_num * self.Ql_dim
        self.hidden_dim = modelType['hidden_dim']
        self.output_dim = modelType['output_dim']
        # Number of hidden layers
        self.layer_num = modelType['layer_num']

        # define feature vectors
        self.QnV = Parameter(nn.init.normal_(torch.zeros(self.Qn_num, self.Qn_dim), mean=0, std=0.5))
        self.QlV = Parameter(nn.init.normal_(torch.zeros(self.Ql_class, self.Ql_dim), mean=0, std=0.5))
        # define sub models
        self.lstm = LSTMCell(self.input_dim, self.hidden_dim, self.layer_num, bias=bias)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim, bias=bias)
        # define batch normal
        self.QnBatch = nn.BatchNorm1d(self.Qn_num)
        # define loss weight
        weight = torch.tensor([1. for i in range(self.output_dim)]) if weight == None else weight
        weight = weight / weight.sum() * weight.shape[0]
        self.weight = weight.reshape(1, -1).to(device)

        # 計算用變數
        Qlvs = []
        s = 0
        for i in self.Ql:
            Qlvs.append(s)
            s += i
        Qlvs = torch.tensor(Qlvs).view(1, 1, -1)
        self.Qlvs = Qlvs.to(device)

    # inputs data transfer to its vector
    def data_trans(self, inputs):
        N, T, _ = inputs.shape
        Qn, Ql = inputs[:, :, :self.Qn_num].type(torch.float), inputs[:, :, self.Qn_num:].type(torch.long)
        a, b, c = Qn.shape
        Qn = self.QnBatch(Qn.reshape(-1, c)).reshape(a, b, c)  # batch normal
        Qn = torch.einsum('ijk, kd->ijkd', Qn, self.QnV)
        Ql += self.Qlvs
        Ql = self.QlV[Ql, :]
        # print(Qn.shape, Ql.shape)
        outputs = torch.cat([Qn, Ql], dim=2)
        return outputs.reshape(N, T, -1)

    def forward(self, x):
        device = self.device
        x = self.data_trans(x)
        # print(x.shape)
        N, T, D = x.shape
        # Initialize hidden state with zeros
        hn = torch.zeros(self.layer_num, N, self.hidden_dim, dtype=torch.float).to(device)
        # Initialize cell state
        cn = torch.zeros(self.layer_num, N, self.hidden_dim, dtype=torch.float).to(device)

        outs = []
        for seq in range(T):
            hn, cn = self.lstm(x[:,seq,:], hn, cn)
            outs.append(hn[-1, :, :].unsqueeze(1))
        outs = torch.cat(outs, dim=1)

        outputs = self.fc(outs)
        outputs = torch.softmax(outputs, dim=2)
        return outputs

    # rmse
    def loss_fun(self, inputs, label):
        device = self.device
        a, b, c = inputs.shape
        N, m = a * b, label.shape[-1]
        inputs = inputs.reshape(N, c)
        label = label.reshape(N, m)
        label = label / (label.sum(dim=1).reshape(N, -1) + 1e-6)
        label = torch.exp(label)
        label = label / (label.sum(dim=1).reshape(N, -1))
        loss = ((inputs - label) ** 2).sum(dim=1)
        loss = torch.sqrt(loss).sum() / N * m
        return loss

    # 亂寫的 ranking loss function
    # def loss_fun(self, inputs, label):
    #     device = self.device
    #     a, b, c = inputs.shape
    #     N, m = a * b, label.shape[-1]
    #     inputs = inputs.reshape(N, c)
    #     label = label.reshape(N, m)
    #     arang = (torch.arange(0, N, 1).reshape(N, -1) * c).to(device)
    #     label = (label + arang).type(torch.long)

    #     e = torch.exp(inputs)
    #     e = torch.log(e / e.sum(dim=1).reshape(N, -1))
    #     ec = e * self.weight * (-1)
    #     outputs = ec.view(-1)[label.view(-1)]
    #     outputs_term1 = outputs.sum() / N

    #     outputs = e.view(-1)[label.view(-1)].reshape(N, -1)
    #     o = []
    #     for i in range(c-1):
    #         o.append(outputs[:, i:i+1] - outputs[:, i+1:i+2])
    #     outputs = torch.cat(o, dim=1)
    #     outputs = outputs * (outputs <= 0)
    #     outputs_term2 = outputs.sum() * (-1) / N
    #     # print(outputs_term1, outputs_term2)
    #     return outputs_term1 + outputs_term2

def train(model, optimizer, data_loader, device, epochs,
        savePath=None, featureType=None, modelType=None, logger=None, re_epochs=0):
    model.to(device)
    for epoch in range(1+re_epochs, epochs+re_epochs+1):
        trainLossList = []
        valLossList = []
        for i, (inputs, label) in enumerate(data_loader):
            # train step
            model.train(True)
            inputs = inputs[:, :, 1:-6].to(device)
            label = label.to(device)
            # trainInputs = inputs[:, :-1, :]   # 測定 model 有無效果需要使用這裡
            # trainLabel = label[:, :-1, :]
            trainInputs = inputs
            trainLabel = label
            # return trainInputs
            outputs = model(trainInputs)
            loss = model.loss_fun(outputs, trainLabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainLoss = loss.item()

            # validation step
            model.train(False)
            with torch.no_grad():
                valInputs = inputs
                valLabel = label[:, -1:, :]

                outputs = model(valInputs)
                outputs = outputs[:, -1:, :]
                # valLoss = ((outputs - valLabel) ** 2).mean().item()
                valLoss = model.loss_fun(outputs, valLabel).item()

                trainLossList.append(trainLoss)
                valLossList.append(valLoss)

        trainLoss = sum(trainLossList) / len(trainLossList)
        valLoss = sum(valLossList) / len(valLossList)
        if logger is not None:
            logger.info(f'epoch: {str(epoch)}/{str(epochs+re_epochs)}, trainLoss: {str(trainLoss)}, valLoss: {str(valLoss)}')

        else:
            print(f'epoch: {str(epoch)}/{str(epochs+re_epochs)}, trainLoss: {str(trainLoss)}, valLoss: {str(valLoss)}')

        if savePath:
            saveDict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch,
                        'featureType': featureType, 'modelType': modelType}
            torch.save(saveDict, savePath)
