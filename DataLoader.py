import torch
import numpy as np
import pickle
import os
import utils

class DataSet:
    def __init__(self, path, idxs):  
        self.path = path
        self.idxs = idxs
        self.data_process()

    def data_process(self):
        datasPath = []
        for i in range(1, 25):
            p = f'dt{str(i)}.pickle'
            datasPath.append(os.path.join(self.path, p))

        datas = []
        for p in datasPath:
            with open(p, 'rb') as f:
                data = pickle.load(f)
                data = torch.tensor(data)
                datas.append(data.unsqueeze(1))

        datas = torch.cat(datas, dim=1)
        datas = datas[self.idxs, :, :]
        datas = utils.inputs_trans(datas.numpy())
        datas[:, :, 1:17] = np.log(datas[:, :, 1:17] + np.e)
        datas[:, :, -3:] = np.log(datas[:, :, -3:] + np.e)
        self.datas = torch.tensor(datas, dtype=torch.float)

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        datas = self.datas[idx, :-1, :]
        labels = self.datas[idx, 1:, 1:17]
        return datas, labels

class testDataSet(DataSet):
    def __init__(self, path, idxs):
        super(testDataSet, self).__init__(path, idxs)

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        datas = self.datas[idx, :, :]
        return datas

def dataLoader(path, idxs, batch_size=1, shuffle=True, train=True):
    if train:
        train_set = DataSet(path, idxs)
    else:
        train_set = testDataSet(path, idxs)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader