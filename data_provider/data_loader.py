import os
import time

import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import datetime
import json

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, df_raw, size=None,
                 features='S',
                 target='OT', scale=False, timeenc=1, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.df_raw = df_raw

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(self.df_raw.columns)
        cols = ['Open', 'High', 'Low', 'Close']
        self.df_raw = self.df_raw[['Date'] + cols]
       
        cols_data = self.df_raw.columns[1:]
        df_data = self.df_raw[cols_data]
    
        if self.scale:
            train_data = df_data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = self.df_raw[['Date']]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)









