import numpy as np
from tqdm import tqdm
import codecs
import re

import torch
import get_info_from_filelist
import sig_proc

################################################################################
#                         CVAE用ドップラーのデータセット                        #
################################################################################
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, datatime_idx, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label
        self.datatime_idx = datatime_idx

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            # print(self.data.shape)
            # print(self.data[idx].shape)
            out_data = self.transform(self.data[idx])
            out_label = int(self.label[idx])
            out_datatime_idx = int(self.datatime_idx[idx])
        else:
            out_data = self.data[idx]
            out_label =  self.label[idx]
            out_datatime_idx = self.datatime_idx[idx]

        return out_data, out_label, out_datatime_idx

################################################################################
#                             データセット作成用の関数                          #
################################################################################
def createMydataset2D(subject_num,filelist, sma_num, wavelet_height, wavelet_width, transform, distance=None, freqs_start=None, freqs_end=None):
    """ピーク前後で分割した時間軸波形をウェーブレット変換したデータを用いてデータセット作成"""

    LABELs = {}
    sub_label = 0

    peaks, min_peaks = get_info_from_filelist.count_peaks_detail(filelist)
    # データかぶらんようにする
    min_peaks -= sma_num-1

    # 訓練データとテストデータの数を決める（8:2）
    numoftrain = int(float(min_peaks * 0.8))
    numoftest = min_peaks - numoftrain
    print(f"\n訓練データ数：{numoftrain}, テストデータ数：{numoftest}\n")

    for subject in peaks.keys():
        if subject not in LABELs:
            LABELs[subject] = sub_label
            sub_label = sub_label + 1
    """
    print("########### 被験者とラベル ##########")
    for subject, label in LABELs.items():
        print(f"被験者：{subject},  ラベル：{label}")"""

    train_datasetlist = np.empty(0)
    test_datasetlist = np.empty(0)
    train_label = []
    test_label = []
    train_datatime_idx = []
    test_datatime_idx = []
    count_train = count_test = 0 #datatime_idx用

    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        #print(f"Input file: {split_fpdata[0]}")
        fname = split_fpdata[0]
    # ----- 色々正規表現で取得
        data_subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        tmp = re.findall('/(.*).npy',fname)[0]
        data_distance = re.findall('[A-Z][a-z]*_(.*)_rec',tmp)[0] #距離　ex)50cm
        #print(data_distance)
        # data_pre = int(re.findall('pre_([0-9]+)_post',fname)[0]) #ピーク前何サンプルとってるか
        is_with_breathing = True if "wo_breath" not in fname else False #呼吸ありか無しか

        if distance == None: #距離混合
            if is_with_breathing and LABELs[data_subject] < subject_num:
                data = np.load(fname)
            # ----- Create Train Dataset
                data_temp = data[:numoftrain,freqs_start:freqs_end,:]
                if freqs_start != None or freqs_end != None:
                    for i in range(data_temp.shape[0]):
                        data_temp[i] = sig_proc.min_max(data_temp[i])
                train_datasetlist = np.append(train_datasetlist, data_temp)
                label = [LABELs[data_subject]]*numoftrain
                train_label.extend(label)
                datatime_idx = list(range(count_train, count_train+numoftrain))
                train_datatime_idx.extend(datatime_idx)
                count_train += numoftrain

            # ----- Create Test Dataset
                data_temp = data[numoftrain+sma_num-1:min_peaks+sma_num-1,freqs_start:freqs_end,:]
                if freqs_start != None or freqs_end != None:
                    for i in range(data_temp.shape[0]):
                        data_temp[i] = sig_proc.min_max(data_temp[i])
                test_datasetlist = np.append(test_datasetlist, data_temp)
                label = [LABELs[data_subject]]*numoftest
                test_label.extend(label)
                #datatime_idx = list(range(numoftrain+sma_num-1,min_peaks+sma_num-1))
                datatime_idx = list(range(count_test,count_test+numoftest))
                test_datatime_idx.extend(datatime_idx)
                count_test += numoftest
            """
            else:
            # ----- Create Test Dataset
                wo_breath_data = np.load(fname)
                npeaks = wo_breath_data.shape[0]
                test_datasetlist = np.append(test_datasetlist, wo_breath_data)
                label = [LABELs[data_subject]]*npeaks
                test_label.extend(label)
                datatime_idx = list(range(min_peaks+sma_num-1, min_peaks+sma_num-1+npeaks))
                test_datatime_idx.extend(datatime_idx)"""

        else: # 距離指定
            if is_with_breathing and data_distance == distance and LABELs[data_subject] < subject_num:
                data = np.load(fname)
            # ----- Create Train Dataset
                data_temp = data[:numoftrain,freqs_start:freqs_end,:]
                if freqs_start != None or freqs_end != None:
                    for i in range(data_temp.shape[0]):
                        data_temp[i] = sig_proc.min_max(data_temp[i])
                train_datasetlist = np.append(train_datasetlist, data_temp)
                label = [LABELs[data_subject]]*numoftrain
                train_label.extend(label)
                #datatime_idx = list(range(numoftrain))
                datatime_idx = list(range(count_train, count_train+numoftrain))
                train_datatime_idx.extend(datatime_idx)
                count_train += numoftrain

            # ----- Create Test Dataset
                data_temp = data[numoftrain+sma_num-1:min_peaks+sma_num-1,freqs_start:freqs_end,:]
                if freqs_start != None or freqs_end != None:
                    for i in range(data_temp.shape[0]):
                        data_temp[i] = sig_proc.min_max(data_temp[i])
                test_datasetlist = np.append(test_datasetlist, data_temp)
                label = [LABELs[data_subject]]*numoftest
                test_label.extend(label)
                #datatime_idx = list(range(numoftrain+sma_num-1,min_peaks+sma_num-1))
                datatime_idx = list(range(count_test, count_test+numoftest))
                test_datatime_idx.extend(datatime_idx)
                count_test += numoftest
            """
            elif data_distance == distance:
            # ----- Create Test Dataset
                wo_breath_data = np.load(fname)
                npeaks = wo_breath_data.shape[0]
                test_datasetlist = np.append(test_datasetlist, wo_breath_data)
                label = [LABELs[data_subject]]*npeaks
                test_label.extend(label)
                datatime_idx = list(range(min_peaks+sma_num-1, min_peaks+sma_num-1+npeaks))
                test_datatime_idx.extend(datatime_idx)"""

    train_datasetlist = np.reshape(train_datasetlist, (-1, wavelet_height, wavelet_width, 1))
    test_datasetlist = np.reshape(test_datasetlist, (-1, wavelet_height, wavelet_width, 1))
    print(f"train_datasetlist.shape : {train_datasetlist.shape}")
    print(f"test_datasetlist.shape : {test_datasetlist.shape}")

    train_label = np.array(train_label)
    test_label = np.array(test_label)
    print(f"train_label.shape : {train_label.shape}")
    print(f"test_label.shape : {test_label.shape}")

    train_datatime_idx = np.array(train_datatime_idx)
    test_datatime_idx = np.array(test_datatime_idx)
    print(f"train_datatime_idx：{train_datatime_idx.shape}")
    print(f"test_datatime_idx：{test_datatime_idx.shape}")

    filelist_fp.close()

    return MyDataset(train_datasetlist.astype(np.float32), train_label, train_datatime_idx, transform), MyDataset(test_datasetlist.astype(np.float32), test_label, test_datatime_idx, transform)
