from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from exp.pattern_model import PatternModel
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import pandas as pd
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.args = args
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):

        if flag == 'train':
            csv_files = [f for f in os.listdir('train') if f.endswith('.csv')]
            df_raw = pd.concat([pd.read_csv(os.path.join('train', f)) for f in csv_files], axis=0)
            # remove 200 elements first in df_raw
            df_raw = df_raw.iloc[200: , :]
            data = Dataset_Custom(df_raw=df_raw, size=[self.args.seq_len, self.args.label_len, self.args.pred_len])
            data_loader = DataLoader(
                data,
                batch_size=self.args.batch_size,
                shuffle=42)
            
        elif flag == 'test':
            csv_files = [f for f in os.listdir('test') if f.endswith('.csv')]
            df_raw = pd.concat([pd.read_csv(os.path.join('test', f)) for f in csv_files], axis=0)
            
            data = Dataset_Custom(df_raw=df_raw, size=[self.args.seq_len, self.args.label_len, self.args.pred_len])
            data_loader = DataLoader(
                data,
                batch_size=self.args.batch_size,
                shuffle=42)
            
        return data, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, train_data, vali_loader, criterion, epoch=0):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
            
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if train_data.scale:
                    shape = outputs.shape
                    outputs = train_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = train_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                folder_result = f'test_results/{epoch}'

                if not os.path.exists(folder_result):
                    os.makedirs(folder_result)

                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if train_data.scale and self.args.inverse:
                        shape = input.shape
                        input = train_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    true_max_idx = np.argmax(true[0, :, -1]) + input[0, :, -1].shape[0]
                    true_min_idx = np.argmin(true[0, :, -1]) + input[0, :, -1].shape[0]

                    pd_max_idx = np.argmax(pred[0, :, -1]) + input[0, :, -1].shape[0]
                    pd_min_idx = np.argmin(pred[0, :, -1]) + input[0, :, -1].shape[0]

                    mae = np.mean(np.abs(pred[0, :, -1] - true[0, :, -1]))

                    diff_x_true = np.concatenate((input[0, :, -2], true[0, :, -2]), axis=0)
                    diff_x_pred = np.concatenate((input[0, :, -2], pred[0, :, -2]), axis=0)

                    name_image = f"{i}_{round(true_max_idx - pd_max_idx, 2)}_{round(true_min_idx - pd_min_idx, 2)}_{round(mae, 2)}"
                    visual(gt, pd, name = os.path.join(folder_result, name_image + '.png'))

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')

        path = "checkpoints" 
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_data, vali_loader, criterion, epoch + 1)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch + 1}.pth')

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def analyze_model(self):
        self.model.load_state_dict(torch.load(os.path.join('models', 'checkpoint_47.pth'), map_location=self.device))
        self.model.eval()

        df_raw_test = pd.read_csv(r"test/exness_xau_usd_h1_2023.csv")
        # df_raw_test.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'volume'] 
        df_stamp = df_raw_test[['Date']]
        data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq='h')
        data_stamp = data_stamp.transpose(1, 0)

        df_raw_test['Date'] = pd.to_datetime(df_raw_test['Date'])
        df = df_raw_test[(df_raw_test['Date'].dt.year == 2023) & (df_raw_test['Date'].dt.month >= 6) & (df_raw_test['Date'].dt.month <= 12)]
        # df = df_raw_test[(df_raw_test['Date'].dt.year == 2024) & (df_raw_test['Date'].dt.month == 1)]
        # df_raw = df_raw_test[['Open', 'High', 'Low', 'volume', 'Close', 'technical_info']]
        self.data = df[['Open', 'High', 'Low', 'volume', 'Close']].values
        self.date_time_2023 = df['Date'].values
        list_mae = []
        total_sl = 0
        total_tp = 0

        total_correction_trend = 0
        list_pattern_predict = []

        for i, data_candle in tqdm.tqdm(enumerate(self.data)):
            mae, list_zigzag_true, list_zigzag_pred, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred = self.inference(i, data_stamp)
            list_mae.append(mae)
            if len(list_zigzag_pred) > 1:
                if len(list_pattern_predict) == 0:
                        pattern_model = PatternModel(list_zigzag_pred, list_zigzag_true, i, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred)
                        list_pattern_predict.append(pattern_model)
                else:
                    last_pattern_model = list_pattern_predict[-1]
                    if last_pattern_model.list_zigzag_pred[0][2] == list_zigzag_pred[0][2]:
                        last_pattern_model.confirm_count += 1
                        last_pattern_model.list_zigzag_true = list_zigzag_true
                        last_pattern_model.list_zigzag_pred = list_zigzag_pred
                        last_pattern_model.index_candle = i
                        last_pattern_model.path_image = path_image
                        last_pattern_model.actual = actual
                        last_pattern_model.pred = pred
                        last_pattern_model.gt = gt
                        last_pattern_model.x_zigzag_data_true = x_zigzag_data_true
                        last_pattern_model.y_zigzag_data_true = y_zigzag_data_true
                        last_pattern_model.x_zigzag_data_pred = x_zigzag_data_pred
                        last_pattern_model.y_zigzag_data_pred = y_zigzag_data_pred

                    else:
                        pattern_model = PatternModel(list_zigzag_pred, list_zigzag_true, i, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred)
                        list_pattern_predict.append(pattern_model)

            # if pd_max_idx is not None or pd_min_idx is not None:
            #     current_close = data_candle[-1]

            #     # price close in pd_max_idx, pd_min_idx
            #     close_pd_max = self.data[pd_max_idx][-1]
            #     close_pd_min = self.data[pd_min_idx][-1]

            #     status = None
            #     if current_close - close_pd_min >= 10 and pd_min_idx <= pd_max_idx:
            #         status = 'sell'
            #     elif close_pd_max - current_close >= 10 and pd_max_idx <= pd_min_idx:
            #         status = 'buy'
            #     else:
            #         status = None

            #     index_sl = None
            #     index_tp = None

            #     if status == 'sell':
            #         for index in range(i, i + self.args.pred_len):
            #             if self.data[index][1] - current_close >= 20:
            #                 index_sl = index
            #             if current_close - self.data[index][2] >= 10:
            #                 index_tp = index
            #     elif status == 'buy':
            #         for index in range(i, i + self.args.pred_len):
            #             if current_close - self.data[index][2] >= 20:
            #                 index_sl = index
            #             if self.data[index][1] - current_close >= 10:
            #                 index_tp = index
                
            #     if status is not None:
            #         if index_sl is not None and index_tp is not None:
            #             if index_sl < index_tp:
            #                 total_sl += 1
            #             elif index_sl > index_tp:
            #                 total_tp += 1
            #         elif index_sl is not None and index_tp is None:
            #             total_sl += 1
            #         elif index_sl is None and index_tp is not None:
            #             total_tp += 1
                    
            #         print(status, index_sl, index_tp)

        count_confirm_pattern = 0     
        count_confirm_corection = 0              
        for pattern_model in list_pattern_predict:
            if pattern_model.confirm_count >= 5:
                count_confirm_pattern += 1
                # if len(pattern_model.list_zigzag_pred) > 1 and len(pattern_model.list_zigzag_true) > 1:
                if pattern_model.list_zigzag_pred[0][2] == pattern_model.list_zigzag_true[0][2]:
                    count_confirm_corection += 1
                    path_image = pattern_model.path_image
                    visual(pattern_model.gt, pattern_model.pred, pattern_model.actual, pattern_model.x_zigzag_data_true, pattern_model.y_zigzag_data_true, pattern_model.x_zigzag_data_pred, pattern_model.y_zigzag_data_pred, self.args.seq_len, path_image)
                else:
                    path_image = pattern_model.path_image.replace('correction', 'fail')
                    if not os.path.exists(os.path.dirname(path_image)): 
                        os.makedirs(os.path.dirname(path_image)) 
                    visual(pattern_model.gt, pattern_model.pred, pattern_model.actual, pattern_model.x_zigzag_data_true, pattern_model.y_zigzag_data_true, pattern_model.x_zigzag_data_pred, pattern_model.y_zigzag_data_pred, self.args.seq_len, path_image)
                # else:
                #     path_image = pattern_model.path_image.replace('correction', 'fail')
                #     if not os.path.exists(os.path.dirname(path_image)): 
                #         os.makedirs(os.path.dirname(path_image))
                #     visual(pattern_model.gt, pattern_model.pred, pattern_model.actual, pattern_model.x_zigzag_data_true, pattern_model.y_zigzag_data_true, pattern_model.x_zigzag_data_pred, pattern_model.y_zigzag_data_pred, self.args.seq_len, path_image)

            print(f'Index: {pattern_model.index_candle} count: {pattern_model.confirm_count} list_type_pred: {pattern_model.list_zigzag_pred} list_type_true: {pattern_model.list_zigzag_true}')
        
        print(f'Count confirm pattern: {count_confirm_pattern}/{len(list_pattern_predict)}')
        print(f'Count confirm corection: {count_confirm_corection}/{count_confirm_pattern}')
    
        # Report 
        # print('Average mae: ', np.mean(list_mae))
        

    def inference(self, index_candle, data_stamp):
        s_end = index_candle
        s_begin = s_end - self.args.seq_len
        r_begin = s_end - self.args.label_len
        r_end = index_candle + self.args.pred_len
        if index_candle <= self.args.seq_len or r_end >= len(self.data):
            return None, [], [], None, None, None, None, None, None, None, None

        batch_x = self.data[s_begin:s_end]
        batch_y = self.data[r_begin:r_end]
        batch_x_mark = data_stamp[s_begin:s_end]
        batch_y_mark = data_stamp[r_begin:r_end]

        actual = self.data[s_begin:r_end][:, -1]

        batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(self.device)
        batch_y = torch.from_numpy(batch_y).float().unsqueeze(0).to(self.device)
        batch_x_mark = torch.from_numpy(batch_x_mark).float().unsqueeze(0).to(self.device)
        batch_y_mark = torch.from_numpy(batch_y_mark).float().unsqueeze(0).to(self.device)

        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

        outputs = np.array([output.detach().cpu().numpy() for output in outputs])
        true = np.array([batch_y.detach().cpu().numpy() for batch_y in batch_y])

        pred = outputs

        folder_path = 'results'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if visual:
            input = np.array([b_x.detach().cpu().numpy() for b_x in batch_x])
            gt = np.concatenate((input[0, :, -1], true[0, :self.args.pred_len, -1]), axis=0)
            pd = np.concatenate((input[0, :, -1], pred[0, :self.args.pred_len, -1]), axis=0)
            true_max_idx = np.argmax(true[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]
            true_min_idx = np.argmin(true[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]

            pd_max_idx = np.argmax(pred[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]
            pd_min_idx = np.argmin(pred[0, :self.args.pred_len, -1]) + input[0, :, -1].shape[0]

            mae = np.mean(np.abs(pred[0, :self.args.pred_len, -1] - true[0, :self.args.pred_len, -1]))

            list_y_candle_true = np.concatenate((input[0, :], true[0, :]), axis=0)
            list_y_candle_pred = np.concatenate((input[0, :], pred[0, :]), axis=0)

            x_zigzag_data_true, y_zigzag_data_true, type_zigzag_data_true = self.analys_zigzag_data(list_y_candle_true)
            x_zigzag_data_pred, y_zigzag_data_pred, type_zigzag_data_pred = self.analys_zigzag_data(list_y_candle_pred)
            
            list_zigzag_true = []
            for i, x in enumerate(x_zigzag_data_true):
                if x > self.args.seq_len:
                    list_zigzag_true.append([x, y_zigzag_data_true[i], type_zigzag_data_true[i]])

            list_zigzag_pred = []
            for i, x in enumerate(x_zigzag_data_pred):
                if x > self.args.seq_len:
                    list_zigzag_pred.append([x, y_zigzag_data_pred[i], type_zigzag_data_pred[i]])

            if len(list_zigzag_true) > 0 and len(list_zigzag_pred) > 0:
                if list_zigzag_true[0] == list_zigzag_pred[0]:
                    correction_trend = 1
                else:
                    correction_trend = 0
            else:
                correction_trend = 0

            name_image = f"{index_candle}_{round(true_max_idx - pd_max_idx, 2)}_{round(true_min_idx - pd_min_idx, 2)}_{correction_trend}_{round(mae, 2)}"
            # visual(gt, pd, actual, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred, os.path.join(folder_path, name_image + '.png'))
            
            path_correction_image = os.path.join(folder_path, 'correction')
            if not os.path.exists(path_correction_image):
                os.makedirs(path_correction_image)
            path_image = os.path.join(path_correction_image, name_image + '.png')

        return mae, list_zigzag_true, list_zigzag_pred, gt, pd, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred
    
    def percent(self, start, stop):
        if start != 0:
            percent = float(((float(start) - float(stop)) / float(start))) * 100
            if percent > 0:
                return percent
            else:
                return abs(percent)
        return 1
    
    def analys_zigzag_data(self, list_y_candle):
        percent_filter = 0.4
        candle_timedata_pass = 3
        last_zigzag = 50
        list_zigzag = []
        for candle_timedata in range(len(list_y_candle)):
            candle_data = list_y_candle[candle_timedata]
            if len(list_zigzag) == 0:
                list_zigzag = [[candle_timedata, candle_data[2],'low'], [candle_timedata, candle_data[1],'high']]
            
            if self.percent(list_zigzag[0][1], list_zigzag[1][1]) < percent_filter:
                if list_zigzag[0][2] == "low":
                    if list_zigzag[0][1] > candle_data[2]:
                        list_zigzag.pop(0)
                        list_zigzag.append([candle_timedata, candle_data[2], "low"])
                    elif list_zigzag[1][1] < candle_data[1]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[1], "high"])
                elif list_zigzag[0][2] == "high":
                    if list_zigzag[0][1] < candle_data[1]:
                        list_zigzag.pop(0)
                        list_zigzag.append([candle_timedata, candle_data[1], "high"])
                    elif list_zigzag[1][1] > candle_data[2]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[2], "low"])

            else:
                if list_zigzag[-1][2] == "low":
                    if list_zigzag[-1][1] > candle_data[2]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[2], "low"])
                    elif (
                        self.percent(list_zigzag[-1][1], candle_data[1])
                        > percent_filter
                    ):
                        if candle_timedata - list_zigzag[-1][0] >= candle_timedata_pass:
                            list_zigzag.append(
                                [candle_timedata, candle_data[1], "high"]
                            )

                elif list_zigzag[-1][2] == "high":
                    if list_zigzag[-1][1] < candle_data[1]:
                        list_zigzag.pop()
                        list_zigzag.append([candle_timedata, candle_data[1], "high"])
                    elif (
                        self.percent(list_zigzag[-1][1], candle_data[2])
                        > percent_filter
                    ):
                        if candle_timedata - list_zigzag[-1][0] >= candle_timedata_pass:
                            list_zigzag.append(
                                [candle_timedata, candle_data[2], "low"]
                            )
                    # elif self.percent(list_zigzag[-1][1], candle_data[2]) < percent_filter and candle_timedata - list_zigzag[-1][0] >= process_config.config[f"{self.time_frame}"]["zigzag"]["candle_timedata_update_zigzag"]:
                    #     list_zigzag.append(
                    #             [candle_timedata, candle_data[2], "low"]
                    #         )

        if len(list_zigzag) == 2:
            if (
                self.percent(list_zigzag[0][1], list_zigzag[1][1])
                >= percent_filter
            ):
                x_data = [x[0] for x in list_zigzag]
                y_data = [x[1] for x in list_zigzag]
                type_data = [x[2] for x in list_zigzag]
            else:
                x_data, y_data, type_data = [], [], []
        else:
            x_data = [x[0] for x in list_zigzag]
            y_data = [x[1] for x in list_zigzag]
            type_data = [x[2] for x in list_zigzag]
                
        list_zigzag = list_zigzag[-last_zigzag:]
        return x_data, y_data, type_data

    def test(self, setting, test=0, epoch=0, criterion=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        total_loss = []
        total_loss_mae = []
        preds = []
        trues = []
        oris = []
        folder_path = './test_results/' + self.args.exp_type + '/' + setting + '/'+'epoch' + str(epoch)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm.tqdm(test_loader)):
                f_dim = -1 if self.args.features == 'MS' else 0
                true = batch_y[:, -self.args.pred_len:, f_dim:].cpu().numpy()
                ori_seq = batch_x.cpu().numpy()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                #invert scale
                output_inverts = np.array([test_data.scaler.inverse_transform(output.detach().cpu().numpy()) for output in outputs])
                batch_y_inverts = np.array([test_data.scaler.inverse_transform(b_y.detach().cpu().numpy()) for b_y in batch_y])

                output_inverts_tensor = torch.tensor(output_inverts, requires_grad=True).float().to(self.device)
                batch_y_inverts_tensor = torch.tensor(batch_y_inverts, requires_grad=True).float().to(self.device)

                # Compute the element-wise absolute difference using PyTorch
                difference_tensor = torch.abs(output_inverts_tensor - batch_y_inverts_tensor)

                # Mean Absolute Error (MAE) in PyTorch
                mae_loss_tensor = difference_tensor.mean(dim=1).mean(dim=0, keepdim=True)
                # Compute loss based on the output columns
                total_loss_mae.append(mae_loss_tensor[0].detach().cpu().numpy())
                pred = outputs
                true = batch_y

                loss = criterion(pred, true)

                total_loss.append(loss.item())

                
                outputs = np.array([test_data.scaler_y.inverse_transform(output.detach().cpu().numpy()) for output in outputs])
                true = np.array([test_data.scaler_y.inverse_transform(b_y.detach().cpu().numpy()) for b_y in batch_y])

                pred = outputs

                preds.append(pred)
                trues.append(true)
                oris.append(ori_seq)
                if visual:
                    if i % 5 == 0:
                        # input = batch_x.detach().cpu().numpy()
                        input = np.array([test_data.scaler_y.inverse_transform(b_x.detach().cpu().numpy()) for b_x in batch_x])
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        if visual:
            preds = np.array(preds)
            trues = np.array(trues)
            oris = np.array(oris)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            oris = oris.reshape(-1, oris.shape[-2], oris.shape[-1])
            print('test shape:', preds.shape, trues.shape)
            # result save
            folder_path = './results/' + self.args.exp_type + '/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open(self.args.exp_type + "_result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'ori.npy', oris)
        total_loss = np.average(total_loss)
        total_loss_mae = np.array(total_loss_mae)
        total_loss_mae = np.mean(total_loss_mae, axis=0)
        return total_loss, total_loss_mae
