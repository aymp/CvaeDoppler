"""
いちいちデータセットを作ってCVAEに通して潜在空間の座標を取得して…ってやってたら時間めっちゃかかるので、
とりあえず訓練済みのCVAEに通した出力をCSV形式で保存したい
本コードは潜在変数3次元、5人訓練したモデルで動作を確認しながら作成
このコードの後は、生体認証の指標（FAR,FRR/FPIR,FNIR）を算出するコードを書く予定（CSV読み込んで…的な）
"""

"""
めっちゃコードクローンしてたので保守性上げるために関数を別ファイルに切り分けました 反省
min_max() -> sig_proc.py
class MyDataset -> cvae_doppler_dataset.py
createMyDataset2D() -> cvae_doppler_dataset.py
"""

import os
import numpy as np
import argparse
#from datetime import datetime
#sns.set(style='darkgrid')

import torch
#import torch.nn as nn
from torchvision import transforms
#from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import model_pool2_125hz
import cvae_doppler_dataset

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

    # 推論モードでcpu使う
    device = torch.device("cpu")
    print(device)

    qy_x = model_pool2_125hz.Qy_x(y_dim=Y_DIM).to(device)
    qz_xy = model_pool2_125hz.Qz_xy(z_dim=Z_DIM, y_dim=Y_DIM).to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset, test_dataset = cvae_doppler_dataset.createMydataset2D(23,FILELIST_IN_PATH, SMA_NUM, WAVELET_HEIGHT, WAVELET_WIDTH, transform, DISTANCE, FREQS_START, FREQS_END)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 学習結果を読み込んで、
    qy_x.load_state_dict(torch.load(DIR_OUT+'qy_x.pth', map_location=lambda storage, loc: storage))
    qz_xy.load_state_dict(torch.load(DIR_OUT+'qz_xy.pth', map_location=lambda storage, loc: storage))
    # 推論モードに設定（Dropout、BN、他の無効化）
    qy_x.eval().to(device)
    qz_xy.eval().to(device)

# ----- train_loaderのデータをCVAEに通した時の出力を調べる
    y_pred = []
    t_list = []
    z_pred = []
    datatime_list = []
    
    for x, t, datatime_idx in train_loader:
        x = x.to(device)
        y = qy_x(x)
        z = qz_xy(x,y)
        t_list.extend(t.numpy())
        y_pred.extend(y.argmax(dim=1).numpy())
        z_pred.extend(z.detach().cpu().numpy())
        datatime_list.extend(datatime_idx.numpy())
    y_pred = np.array(y_pred)
    t_array = np.array(t_list)
    z_pred = np.array(z_pred)
    datatime_array = np.array(datatime_list)

    # 訓練済みデータにはフラグを立てたい
    # t_array<Y_DIMならTrueを、そうでないならFalseを返すような列を用意
    trained_flag_array = t_array < Y_DIM

    # CSVファイルに入れていく
    output_fp = open(DIR_OUT+"cvae_output.csv", 'w')
    for label_i, latent_i, datatime_i, label_pred, flag in zip(t_array, z_pred, datatime_array, y_pred, trained_flag_array):
        output_fp.write(str(label_i))
        output_fp.write(",")
        output_fp.write(str(label_pred))
        output_fp.write(",")
        for latent_val in latent_i:
            output_fp.write(str(latent_val))
            output_fp.write(",")
        output_fp.write(str(datatime_i))
        output_fp.write(",")
        output_fp.write(str(flag))
        output_fp.write("\n")
    output_fp.close()

# ----- 今度はtest_loaderにあるデータ
    y_pred = []
    t_list = []
    z_pred = []
    datatime_list = []

    for x, t, datatime_idx in test_loader:
        x = x.to(device)
        y = qy_x(x)
        z = qz_xy(x,y)
        t_list.extend(t.numpy())
        y_pred.extend(y.argmax(dim=1).numpy())
        z_pred.extend(z.detach().cpu().numpy())
        datatime_list.extend(datatime_idx.numpy())

    y_pred = np.array(y_pred)
    t_array = np.array(t_list)
    z_pred = np.array(z_pred)
    datatime_array = np.array(datatime_list)
    trained_flag_array = np.full_like(t_array,False)

    # 先ほどのCSVファイルに追記する　これで全データがCSVに書き込まれるはず
    output_fp = open(DIR_OUT+"cvae_output.csv", 'a')
    for label_i, latent_i, datatime_i, label_pred in zip(t_array, z_pred, datatime_array, y_pred):
        output_fp.write(str(label_i))
        output_fp.write(",")
        output_fp.write(str(label_pred))
        output_fp.write(",")
        for latent_val in latent_i:
            output_fp.write(str(latent_val))
            output_fp.write(",")
        output_fp.write(str(datatime_i))
        output_fp.write(",")
        output_fp.write(str(flag))
        output_fp.write("\n")
    output_fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--distance", default='75cm')
    parser.add_argument("--y_dim", type=int, default=5)
    parser.add_argument("--z_dim", type=int, default=3)
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