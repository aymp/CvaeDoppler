import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pywt
from matplotlib.font_manager import FontProperties
from spectrum import *
from scipy import signal
import codecs
import os
import re
from tqdm import tqdm
import argparse
import statistics

sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

def draw3d(data, x_axis, y_axis, cb_min, cb_max):  # cb_min,cb_max:カラーバーの下端と上端の値
    """PSD描画用の関数"""
    Y, X = np.meshgrid(y_axis, x_axis)
    # 図を描くのに何色用いるか（大きくすると重くなる。小さくすると荒くなる。）
    div = 30.0
    delta = (cb_max - cb_min) / div
    interval = np.arange(cb_min, abs(cb_max) * 2 + delta, delta)[0:int(div) + 1]
    # plt.rcParams["font.size"] = 3
    plt.contourf(X, Y, data, interval)

def decide_order_AIC(filelist_path, win_len=32, order_num=32):
    """　AICが最小となるYule-Walkerの次数を読み込みデータすべてに対し探索し、その最頻値を返す """
    filelist_fp = codecs.open(filelist_path, 'r')
    select_order_list = []
    # ----- ファイル読み込み
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        split_data = np.load(fname)
        npeaks = split_data.shape[0]

        for peak_idx in range(npeaks):
            iq_diff_2nd = split_data[peak_idx][6]
            for i in range(len(iq_diff_2nd)):
                if (i + win_len) <= len(iq_diff_2nd):
                    target_data = iq_diff_2nd[i:(i + win_len)]
                    order = np.arange(1,order_num)
                    rho = [aryule(target_data * window_blackman(win_len), i , norm='biased')[1] for i in order]
                    AIC_n = AIC(win_len,rho,order)
                    """ print(AIC_n)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(order,AIC(win_len,rho,order),label='AIC')
                    plt.show() """
                    select_order = list(AIC_n).index(min(AIC_n))+1
                    select_order_list.append(select_order)
                    #plt.hist(select_order_list)
                    #plt.show()
    order_mode = statistics.mode(select_order_list)
    fig = plt.figure()
    plt.hist(select_order_list,bins=31)
    fig_path = 'AIC_'+filelist_path.replace('0_FileList/','').replace('txt','png')
    fig.savefig(fig_path)
    return order_mode

def decide_order_cAIC(filelist_path, win_len=32, order_num=30):
    """　cAICが最小となるYule-Walkerの次数を読み込みデータすべてに対し探索し、その最頻値を返す """
    filelist_fp = codecs.open(filelist_path, 'r')
    select_order_list = []
    # ----- ファイル読み込み
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        split_data = np.load(fname)
        npeaks = split_data.shape[0]

        for peak_idx in range(npeaks):
            iq_diff_2nd = split_data[peak_idx][6]
            # for i in range(int(len(data_filt))):
            for i in range(len(iq_diff_2nd)):
                if (i + win_len) <= len(iq_diff_2nd):
                    target_data = iq_diff_2nd[i:(i + win_len)]
                    order = np.arange(1,order_num)
                    rho = [aryule(target_data * window_blackman(win_len), i , norm='biased')[1] for i in order]
                    AIC_n = AICc(win_len,rho,order)
                    """ print(AIC_n)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(order,AICc(win_len,rho,order),label='AICc')
                    plt.show() """
                    select_order = list(AIC_n).index(min(AIC_n))+1
                    select_order_list.append(select_order)
                    #plt.hist(select_order_list,bins=30)
                    #plt.show()

    order_mode = statistics.mode(select_order_list)
    fig = plt.figure()
    plt.hist(select_order_list,bins=29)
    fig_path = 'cAIC'+filelist_path.replace('0_FileList/','').replace('txt','png')
    fig.savefig(fig_path)
    
    return order_mode

def yule_walker_AIC(data, win_len=32, order_num=32, nfft=256):
    """ Yule-Waller法 
    http://thomas-cokelaer.info/software/spectrum/html/contents.html
    WIN_LEN = 32 #ウィンドウ幅
    ORDER = 12 #ARモデル次数
    """
    data_temp = []
    # for i in range(int(len(data_filt))):
    for i in range(len(data)):
        if (i + win_len) <= len(data):
            target_data = data[i:(i + win_len)]

            order = np.arange(1,order_num)
            rho = [aryule(target_data * window_blackman(win_len), i , norm='biased')[1] for i in order]
            AIC_n = AIC(win_len,rho,order)
            #print(AIC_n)
            select_order = list(AIC_n).index(min(AIC_n))+1
            #print(f"select_order:{select_order}")
            AR, P, k = aryule(target_data * window_blackman(win_len), select_order, norm='biased') #ブラックマン窓
            #print(rho)
            """fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(order,AIC(win_len,rho,order),label='AIC')
            plt.show()"""
            # totalscal点で解析
            PSD = arma2psd(AR,NFFT=nfft)
            temp_PSD = PSD[len(PSD):int(len(PSD) / 2):-1]   # PSDの半分だけ
            data_temp.append(temp_PSD)
    data_temp = np.array(data_temp).T
    #logを取る
    yule_data = np.log(data_temp)
    return yule_data

def yule_walker_cAIC(data, win_len=32, order_num=30, nfft=256):
    """ Yule-Waller法 
    http://thomas-cokelaer.info/software/spectrum/html/contents.html
    WIN_LEN = 32 #ウィンドウ幅
    ORDER = 12 #ARモデル次数
    """
    data_temp = []
    # for i in range(int(len(data_filt))):
    for i in range(len(data)):
        if (i + win_len) <= len(data):
            target_data = data[i:(i + win_len)]

            order = np.arange(1,order_num)
            rho = [aryule(target_data * window_blackman(win_len), i , norm='biased')[1] for i in order]
            AIC_n = AICc(win_len,rho,order)
            #print(AIC_n)
            select_order = list(AIC_n).index(min(AIC_n))+1
            #print(f"select_order:{select_order}")
            AR, P, k = aryule(target_data * window_blackman(win_len), select_order, norm='biased') #ブラックマン窓
            #print(rho)
            """fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(order,AIC(win_len,rho,order),label='AIC')
            plt.show()"""
            # totalscal点で解析
            PSD = arma2psd(AR,NFFT=nfft)
            temp_PSD = PSD[len(PSD):int(len(PSD) / 2):-1]   # PSDの半分だけ
            data_temp.append(temp_PSD)
    data_temp = np.array(data_temp).T
    #logを取る
    yule_data = np.log(data_temp)
    return yule_data

def yule_walker(data, win_len=32, order=32, nfft=256):
    #http://thomas-cokelaer.info/software/spectrum/html/contents.html
    data_temp = []
    # for i in range(int(len(data_filt))):
    for i in range(len(data)):
        if (i + win_len) <= len(data):
            target_data = data[i:(i + win_len)]
            #AR, P, k = aryule(target_data * window_hamming(win_len), order) #ハミング窓
            AR, P, k = aryule(target_data * window_blackman(win_len), order) #ブラックマン窓
            # totalscal点で解析
            PSD = arma2psd(AR, NFFT=nfft)
            temp_PSD = PSD[len(PSD):int(len(PSD) / 2):-1]   # PSDの半分だけ
            data_temp.append(temp_PSD)
    data_temp = np.array(data_temp).T
    #logを取る
    yule_data = np.log(data_temp)
    return yule_data

def sample():
    sampling_rate = 1024
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    f1 = 100
    f2 = 200
    f3 = 300
    data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                        [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
                        lambda t: np.sin(2 * np.pi * f3 * t) + np.sin(2 * np.pi * f1 * t)])

    ### CWT
    wavename = 'cgau8'
    totalscal = 256
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    #scales = np.arange(1, 255)#cparam / np.arange(totalscal, 1, -1)
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)

    #http://thomas-cokelaer.info/software/spectrum/html/contents.html
    ### AR model
    WIN_LEN = 32 #ウィンドウ幅
    ORDER = 12 #ARモデル次数

    # ユールウォーカー法
    plt_data_temp = []
    # for i in range(int(len(data_filt))):
    for i in range(len(data)):
        if (i + WIN_LEN) <= len(data):
            target_data = data[i:(i + WIN_LEN)]
            #AR, P, k = aryule(target_data * window_hamming(WIN_LEN), ORDER) #ハミング窓
            AR, P, k = aryule(target_data * window_blackman(WIN_LEN), ORDER) #ブラックマン窓
            # totalscal点で解析
            PSD = arma2psd(AR, NFFT=totalscal)
            temp_PSD = PSD[len(PSD):int(len(PSD) / 2):-1]   # PSDの半分だけ
            plt_data_temp.append(temp_PSD)
    plt_data_temp = np.array(plt_data_temp)
    print(plt_data_temp.shape)
    #logを取る
    yule_plt_data = np.log(plt_data_temp)

    #バーグ法、※次数を上げすぎると ValueErrorが出る？
    plt_data_temp = []
    # for i in range(int(len(data_filt))):
    for i in range(len(data)):
        if (i + WIN_LEN) <= len(data):
            target_data = data[i:(i + WIN_LEN)]
            AR, P, k = arburg(target_data * window_blackman(WIN_LEN), ORDER) #ブラックマン窓
            # totalscal点で解析
            PSD = arma2psd(AR, NFFT=totalscal)
            temp_PSD = PSD[len(PSD):int(len(PSD) / 2):-1]   # PSDの半分だけ
            plt_data_temp.append(temp_PSD)
    plt_data_temp = np.array(plt_data_temp)
    #logを取る
    pburg_plt_data = np.log(plt_data_temp)

    ### STFT
    f, ti, Zxx = signal.stft(data, fs=sampling_rate, nperseg=totalscal)

    #比較
    plt.figure(figsize=(8, 6))
    plt.title(u"300Hz 200Hz 100Hz Time spectrum")

    ax = plt.subplot(511)
    plt.plot(t, data)
    ax.set_title(u"Input")
    plt.xlim(t[0], t[-1])

    ax = plt.subplot(512)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    ax.set_title(u"CWT")

    ax = plt.subplot(513)
    draw3d(np.array(yule_plt_data), np.arange(yule_plt_data.shape[0]), np.arange(yule_plt_data.shape[1]) * (sampling_rate / totalscal), np.min(yule_plt_data), np.max(yule_plt_data))
    ax.set_title(u"Yule-Walker")

    ax = plt.subplot(514)
    draw3d(np.array(pburg_plt_data), np.arange(pburg_plt_data.shape[0]), np.arange(pburg_plt_data.shape[1]) * (sampling_rate / totalscal), np.min(pburg_plt_data), np.max(pburg_plt_data))
    ax.set_title(u"Burg")

    ax = plt.subplot(515)
    plt.contourf(ti, f, np.abs(Zxx), vmin=0, vmax=2 * np.sqrt(2))
    ax.set_title(u"STFT")
    plt.xlabel(u"time(s)")

    plt.tight_layout()
    plt.show()
    print("exit")

def main(args):
    # ----- データ入出力するディレクトリ
    DIR_IN = args.dir_in
    DIR_OUT = args.dir_out
    FILELIST_DIR = args.filelist_dir
    FLISTNAME_IN = args.flistname_in
    FLISTNAME_OUT = args.flistname_out

    WIN_LEN = args.win_len
    ORDER = args.order

    if not os.path.exists(FILELIST_DIR):
        os.mkdir(FILELIST_DIR)
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)
    
    #データセットをファイルリストとしてテキストファイルに書き込みたい。
    FILELIST_OUT_PATH = FILELIST_DIR + FLISTNAME_OUT
    with open(FILELIST_OUT_PATH, mode='w') as _f:
        pass #ここではファイル作るだけ

    #ファイルリスト読み込み
    FILELIST_IN_PATH = FILELIST_DIR + FLISTNAME_IN
    filelist_fp = codecs.open(FILELIST_IN_PATH, 'r')

# ----- AIC最小となるYule-Walkerの次数(order)の最頻値を取得
    #order = decide_order(FILELIST_IN_PATH)
    print(ORDER)

# ----- ファイル読み込み
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        split_data = np.load(fname)
        npeaks = split_data.shape[0]
        yule_walker_data = np.empty(0)

    # ----- iq差分の2段階増幅だけYule-Walker。split_data[i]の中には順にppg,i1,i2,q1,q2,iqdiff1,iqdiff2が入っている。
        for peak_idx in range(npeaks):
            iq_diff_2nd = split_data[peak_idx][6]
            #yule_walker_temp = yule_walker_cAIC(iq_diff_2nd)
            yule_walker_temp = yule_walker(iq_diff_2nd,win_len=WIN_LEN,order=ORDER)
            yule_walker_data = np.append(yule_walker_data,yule_walker_temp)
            
    # ----- npy形式で保存
        yule_walker_data = np.reshape(yule_walker_data, (npeaks,yule_walker_temp.shape[0],yule_walker_temp.shape[1]))
        #yule_walker_data = np.reshape(yule_walker_data, (npeaks,yule_walker_temp.shape))
        print(yule_walker_data.shape)
        data_out_path = fname.replace(DIR_IN, DIR_OUT)
        np.save(data_out_path, yule_walker_data)
    # ----- ファイルリストにファイル名保存
        with open(FILELIST_OUT_PATH, mode='a') as f:
            f.write(data_out_path + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_in", default='3A_SplitData/')
    parser.add_argument("--dir_out", default='5A_YuleWalkerData_cAICmix/')
    parser.add_argument("--filelist_dir", default='0_FileList/')
    parser.add_argument("--flistname_in", default='3A_Split.txt')
    parser.add_argument("--flistname_out", default='5A_YuleWalker_cAICmix.txt')

    parser.add_argument("--win_len", type=int, default=32)
    parser.add_argument("--order", type=int, default=12)

    args = parser.parse_args()
    main(args)
    #sample()
    #decide_order('0_FileList/3A_Split.txt')
    """
    print(f"c-AIC{decide_order_cAIC('0_FileList/3A_Split.txt')}")
    print(f"AIC{decide_order_AIC('0_FileList/3A_Split.txt')}")
    print(f"AIC{decide_order_AIC('0_FileList/3A_Split_500Hz.txt')}")
    print(f"c-AIC{decide_order_cAIC('0_FileList/3A_Split_500Hz.txt')}")"""