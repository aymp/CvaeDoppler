import os
import sys
#sys.path.append('/home/manabe/.local/lib/python3.8/site-packages') #GPU使うとき無理やりパス通す←マウントオプションの変更で解決済。
#print(sys.version)
#print(sys.path)
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex, ListedColormap, Normalize
from matplotlib import ticker
import codecs
import re
import math
from tqdm import tqdm
import argparse
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import subprocess
#from datetime import datetime
#sns.set(style='darkgrid')

import torch
import torch.nn as nn
from torchvision import transforms
#from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import get_info_from_filelist
import model_pool2_125hz

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

def min_max(x, axis=None):
    """0-1の範囲に正規化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

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
                        data_temp[i] = min_max(data_temp[i])
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
                        data_temp[i] = min_max(data_temp[i])
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
                        data_temp[i] = min_max(data_temp[i])
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
                        data_temp[i] = min_max(data_temp[i])
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

def createNoiseDataset(height, width, transform):
    label = [100]*10000
    datatime_idx = [0]*10000
    label = np.array(label)
    datatime_idx = np.array(datatime_idx)
    noise_dataset = np.random.rand(10000,height,width)
    for i in range(noise_dataset.shape[0]):
        noise_dataset[i] = min_max(noise_dataset[i])
    noise_dataset = np.reshape(noise_dataset,(10000,height,width,1))

    return MyDataset(noise_dataset.astype(np.float32),label,datatime_idx,transform)

def main(args):
# ----- パラメータ設定
    DISTANCE = args.distance
    DIST2 = '75cm'
    DIST3 = '1m'
    Z_DIM = args.z_dim
    K = Y_DIM = args.y_dim
    IS_SUPERVISED = args.disable_supervised
    EPOCH_NUM = args.epoch_num #200
    TAU = args.tau # 温度
    BETA = args.beta #7 kl_gaussの係数
    ALPHA = args.alpha #1000 Xentの係数
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    LR = args.lr # 学習率
    SMA_NUM = args.sma_num
    WAVELET_HEIGHT = args.wavelet_height
    WAVELET_WIDTH = args.wavelet_width
    PLOT_RECON = args.plot_recon
    FREQS_START = args.freqs_start
    FREQS_END = args.freqs_end
    FS = args.fs
    POOL = args.pool
    YULE = args.yule
    Y_DIM_ALL = args.y_dim_all

# ----- データ入出力するディレクトリ
    PARENT_DIR_OUT = args.pardir
    if DISTANCE == None:
        CHILD_DIR_OUT = 'sma'+str(SMA_NUM)+'_zdim'+str(Z_DIM)+'_pool'+str(POOL)+'_epoch'+str(EPOCH_NUM)+'_ydim'+str(Y_DIM)+'/'
    else:
        #CHILD_DIR_OUT = 'sma'+str(SMA_NUM)+'_zdim'+str(Z_DIM)+'_pool'+str(POOL)+'_'+DISTANCE+'_epoch'+str(EPOCH_NUM)+'_ydim'+str(Y_DIM)+'/'
        CHILD_DIR_OUT = 'sma'+str(SMA_NUM)+'_zdim'+str(Z_DIM)+'_pool'+str(POOL)+'_'+DISTANCE+'_epoch'+str(EPOCH_NUM)+'_ydim'+str(Y_DIM)+'/'

    DIR_OUT = PARENT_DIR_OUT + CHILD_DIR_OUT

    FILELIST_DIR = args.filelist_dir
    FLISTNAME_IN = args.flistname_in
    FILELIST_IN_PATH = FILELIST_DIR + FLISTNAME_IN

    if not os.path.exists(PARENT_DIR_OUT):
        os.mkdir(PARENT_DIR_OUT)
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)

    if PLOT_RECON and not os.path.exists(DIR_OUT+'figs/'):
        os.mkdir(DIR_OUT+'figs/')

    if not os.path.exists(FILELIST_DIR):
        os.mkdir(FILELIST_DIR)

    ### 12/04追記 推論モードでcpu使う ###
    device = torch.device("cpu")
    print(device)

    #px_z = model_pool2_125hz.Px_z(z_dim=Z_DIM).to(device)
    #pz_y = model_pool2_125hz.Pz_y(z_dim=Z_DIM, y_dim=Y_DIM).to(device)
    qy_x = model_pool2_125hz.Qy_x(y_dim=Y_DIM).to(device)
    qz_xy = model_pool2_125hz.Qz_xy(z_dim=Z_DIM, y_dim=Y_DIM).to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset, test_dataset = createMydataset2D(Y_DIM_ALL,FILELIST_IN_PATH, SMA_NUM, WAVELET_HEIGHT, WAVELET_WIDTH, transform, DISTANCE, FREQS_START, FREQS_END)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #学習結果を読み込んで、
    #px_z.load_state_dict(torch.load(DIR_OUT+'px_z.pth', map_location=lambda storage, loc: storage))
    #pz_y.load_state_dict(torch.load(DIR_OUT+'pz_y.pth', map_location=lambda storage, loc: storage))
    qy_x.load_state_dict(torch.load(DIR_OUT+'qy_x.pth', map_location=lambda storage, loc: storage))
    qz_xy.load_state_dict(torch.load(DIR_OUT+'qz_xy.pth', map_location=lambda storage, loc: storage))
    #推論モードに設定（Dropout、BN、他の無効化）.to(device)は12/04追加。
    #px_z.eval().to(device)
    #pz_y.eval().to(device)
    qy_x.eval().to(device)
    qz_xy.eval().to(device)

    #訓練データの分布の中心座標を求める．


    y_pred = []
    t_list = []
    z_pred = []
    
    for x, t, _datatime_idx in train_loader:
        x = x.to(device)
        y = qy_x(x)
        z = qz_xy(x,y)
        t_list.extend(t.numpy())
        y_pred.extend(y.argmax(dim=1).numpy())
        z_pred.extend(z.detach().cpu().numpy())
    y_pred = np.array(y_pred)
    t_array = np.array(t_list)
    z_pred = np.array(z_pred)
    print(z_pred.shape)

# ----- TRAINED
    z_trained_avg = {}
    for i in range(Y_DIM):
        z_trained = z_pred[t_array == i]
        z_trained_avg[i] = z_trained.sum(axis=0)/z_trained.shape[0]
        print(i,len(z_trained),z_trained_avg[i])
    #print(f"{}{}")

# ----- ADDITIONAL SUBJECT
    z_add_avg = {}
    for i in range(Y_DIM):
        z_add = z_pred[t_array >= Y_DIM]
        y_add_pred = y_pred[t_array >= Y_DIM]
        z_add = z_add[y_add_pred == i]
        z_add_avg[i] = z_add.sum(axis=0)/z_add.shape[0]
        print(i,len(z_add),z_add_avg[i])
    # ----- DISTANCE between each coordinate and trained-data's centroid
        z_add_vector = z_add - z_trained_avg[i] #broadcast
        z_add_dist = np.linalg.norm(z_add_vector, ord=2, axis=1)
        print(f"Add_Avg{i}:{z_add_dist.sum(axis=0)/z_add_dist.shape[0]}")

    #acc = accuracy_score(t_array,y_pred)
    #print(acc)

    y_pred = []
    t_list = []
    z_pred = []
    for x, t, _datatime_idx in test_loader:
        x = x.to(device)
        y = qy_x(x)
        z = qz_xy(x,y)
        t_list.extend(t.numpy())
        y_pred.extend(y.argmax(dim=1).numpy())
        z_pred.extend(z.detach().cpu().numpy())
    y_pred = np.array(y_pred)
    t_array = np.array(t_list)
    z_pred = np.array(z_pred)
    #acc = accuracy_score(t_array,y_pred)
    #print(f"{DIST2}_acc:{acc}")

# ----- TEST DATA (TRAINED SUBJECT)
    z_trained_test_avg = {}
    for i in range(Y_DIM):
        z_test = z_pred[t_array < Y_DIM]
        y_test_pred = y_pred[t_array < Y_DIM]
        z_test = z_test[y_test_pred == i]
        z_trained_test_avg[i] = z_test.sum(axis=0)/z_test.shape[0]
        print(i,len(z_test),z_trained_test_avg[i])
    # ----- DISTANCE between each coordinate and trained-data's centroid
        z_trained_vector = z_test - z_trained_avg[i] #broadcast
        z_trained_dist = np.linalg.norm(z_trained_vector, ord=2, axis=1)
        print(f"Trained_Avg{i}:{z_trained_dist.sum(axis=0)/z_trained_dist.shape[0]}")

# ----- CALC L2 NORM
    for i in range(Y_DIM):
        trained = z_trained_avg[i] - z_trained_test_avg[i]
        print(i,np.linalg.norm(trained, ord=2))
        additional = z_trained_avg[i] - z_add_avg[i]
        print(i,np.linalg.norm(additional, ord=2))

        

    cm = confusion_matrix(t_array,y_pred)
    fig = plt.figure(figsize=(4,4),dpi=300)
    fig.tight_layout()
    ax = plt.subplot()
    fig.subplots_adjust(top=0.85,bottom=0.05)
    fig = sns.heatmap(cm,annot=True,fmt="d",cmap='Blues',square=True,cbar=False,annot_kws={"fontsize":6})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    #plt.savefig(DIR_OUT+DIST2+str(acc)+".png")
    #plt.savefig(DIR_OUT+DIST2+str(acc)+".svg")
    #subprocess.call('inkscape -M '+DIR_OUT+DIST2+str(acc)+'.emf '+DIR_OUT+DIST2+str(acc)+'.svg',shell=True)
    plt.close()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--distance", default='75cm')
    parser.add_argument("--y_dim", type=int, default=5)
    parser.add_argument("--z_dim", type=int, default=2)
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.5)#温度
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sma_num", type=int, default=5)

    parser.add_argument("--fs", type=int, default=125)
    parser.add_argument("--pool", type=int, default=2)

    parser.add_argument("--freqs_start", type=int, default=3)
    parser.add_argument("--freqs_end", type=int, default=51)

    parser.add_argument("--disable_supervised", action='store_false')#デフォTrue
    parser.add_argument("--plot_recon", action='store_true')#デフォFalse
    parser.add_argument("--yule", action='store_true')

    parser.add_argument("--wavelet_height", type=int, default=48)
    parser.add_argument("--wavelet_width", type=int, default=74)

    parser.add_argument("--pardir", default='6_ClusteringResults/')
    parser.add_argument("--filelist_dir", default='0_FileList/')
    parser.add_argument("--flistname_in", default='4C_LogSma5Wavelet_125Hz.txt')

    # additional
    parser.add_argument("--y_dim_all", type=int, default=6)


    args = parser.parse_args()
    main(args)
