import numpy as np
import matplotlib.pyplot as plt
import codecs
import os
from tqdm import tqdm
import argparse

import sig_proc

def min_max(x, axis=None):
    """0-1の範囲に正規化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def data_distribution(data, len_data, peak_times, peak_vals, pre_sample_num, post_sample_num):
    """
    ピークの前pre_sec[s], 後ろpost_sec[s]のデータを連結させた配列を返す。分割は他で...
    
    Parameters
    ----------
    data : ndarray
   5    入力データ(1行目ppg, 2行目iq_diff_1st, 3行目iq_diff_2nd想定)。
   6len_data : int
        入力データのサンプル数。ピーク後何秒とるか？で使う。
    peak_times : list
        ピーク列(インデックス)。
    peai_vals : list
        ピーク列(値)。
    pre_sample_num : int
        ピーク前何サンプルとるか。
    post_sample_num : int
        ピーク後何サンプルとるか。

    Returns
    -------
    data_out : ndarray
        ピーク前後で分割されたデータがピーク数分入った出力データ。
    npeaks : int
        ピーク数(条件によって入力と変わる可能性あるので)
    len_per_one : int
        分割データ1つ分のサンプル数。
    peak_times : list
        入力時と変わる可能性あるのでReturn。
    peak_vals : list
        同上。
    """
    THRESHOLD = 0.0 # これよりピーク値小さいものは異常値として削除。
    for idx, val in enumerate(peak_vals):
        if val < THRESHOLD:
            peak_times.pop(idx)
            peak_vals.pop(idx)

    #ピーク前に必要数要素確保
    if peak_times[0] <= pre_sample_num:
        peak_times.pop(0)
        peak_vals.pop(0)

    #ピーク後に必要数要素確保
    if peak_times[-1] >= len_data - post_sample_num:
        peak_times.pop(-1)
        peak_vals.pop(-1)

    npeaks = len(peak_times) #ピーク数
    len_per_one = int(pre_sample_num+post_sample_num) #分割データ一つ分のサンプル数（データ長）

    data_out = np.empty(0)
    for idx in peak_times:
        data_temp = data[:,idx-pre_sample_num:idx+post_sample_num]
        data_out = np.append(data_out,data_temp)

    # 追記　3次元配列にする
    data_out = np.reshape(data_out,(-1,7,len_per_one))
        
    return data_out, npeaks, len_per_one, peak_times, peak_vals

def main(args):
    """ ピーク検出と、分割データセットの作成 """
# ----- パラメータ
    fs = args.fs # sampling rate
    pre_sec = args.pre_sec#[0.5, 0.52, 0.48]
    post_sec = args.post_sec#[0.1, 0.08, 0.12]
    pre_sample_num = int(pre_sec*fs)
    post_sample_num = int(post_sec*fs)
    subject_num = args.subject_num
    da = 2*args.shift_num*args.shift_width+1 # データ何倍に水増ししてるか

    fig, axes = plt.subplots(subject_num*da, 6) #breath & withoutt
    fig2, axes2 = plt.subplots(subject_num*da, 6)

# ----- データ入出力に用いるディレクトリ
    filelist_dir = args.filelist_dir
    if not os.path.exists(filelist_dir):
        os.mkdir(filelist_dir)
    dir_out = args.dir_out
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    #ファイルリスト読み込み
    filelist =  filelist_dir+args.flistname_in
    filelist_fp = codecs.open(filelist, 'r')

    #データセットをファイルリストとしてテキストファイルに書き込みたい。
    filelist_out = filelist_dir+args.flistname_out
    with open(filelist_out, mode='w') as f:
        pass #何もしない。データセット作るたびにそのcsvファイル名を追記。

# ----- ファイル読み込み
    for idx, filename in tqdm(enumerate(filelist_fp)):
        fname = filename.rstrip('\r\n')
        #data = genfromfile(fname,7)
        data = np.load(fname)
        ppg = data[0]
        # i_1st = data[1]
        # i_2nd = data[2]
        # q_1st = data[3]
        # q_2nd = data[4]
        # iq_diff_1st = data[5]
        # iq_diff_2nd = data[6]
        len_data = data.shape[1]

        fname = fname.replace(args.dir_in, '').replace('.npy','')

    # ----- 時系列探索でppgピーク検出　&　分割。
        peak_times, peak_vals = sig_proc.peak_search(min_max(ppg), fs)
        for shift in range(-args.shift_num*args.shift_width, args.shift_num*args.shift_width+1, args.shift_width):
            # 必要な部分だけ集めたデータdata_dist。あとから一拍ずつ分割。
            data_dist, npeaks, len_per_one, peak_times, peak_vals = data_distribution(data, len_data, peak_times, peak_vals, pre_sample_num+shift, post_sample_num-shift)
            print(data_dist.shape)
            print("ピーク数：" + str(npeaks))
            """
            # データ分割。
            for k in range(npeaks):
                ch_split = []
                for i in range(7):
                    ch_split.append([])
                for j in range(len_per_one):
                    for i in range(7):
                        ch_split[i].append(data_dist[i][len_per_one*k + j])
                # csvに保存
                filename_out = dir_out+'/'+fname+'_pre_'+str(int(pre_sec[l]*fs))+'_post_'+str(int(post_sec[l]*fs))+'_'+str(k)+'.csv'
                np.savetxt(filename_out, np.vstack(
                    [ch_split[0], ch_split[1], ch_split[2], ch_split[3], ch_split[4], ch_split[5], ch_split[6]]), delimiter=',')
                # ファイルリストにファイル名保存
                with open(filelist_out, mode='a') as f:
                    f.write(filename_out + ',0,' + str(len_per_one-1) + '\n')         """  
        # ----- 分割ndarrayをバイナリで保存
            filename_out = args.dir_out+fname+'_pre_'+str(pre_sample_num+shift)+'_post_'+str(post_sample_num-shift)+'.npy'
            np.save(filename_out,data_dist)
            # ファイルリストにファイル名保存
            with open(filelist_out, mode='a') as f:
                f.write(filename_out+'\n')

            # 最初の部分をそれぞれ分割してプロット
            """
            ch_dist = []
            for i in range(7):
                ch_dist.append([])
            for i in range(len_per_one):
                for j in range(7):
                    ch_dist[j].append(data_dist[j][i])"""
            dist_plot = data_dist[0]

            axes2[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].plot(dist_plot[0], label = 'ppg', linewidth=0.5)
            axes3 = axes2[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].twinx()
            axes3.plot(dist_plot[2], label = 'i_2nd', color='orange', linewidth=0.5)
            #axes2[idx//5][idx%5].plot(ch_dist[1], label = 'iq_diff_1st', linewidth=0.5)
            axes4 = axes2[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].twinx()
            axes4.plot(dist_plot[4], label = 'q_2nd', color='green', linewidth=0.5)
            #axes2[idx//5][idx%5].plot(ch_dist[2], label = 'iq_diff_2nd', linew6dth=0.5)
            axes2[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].tick_params(labelsize=6)
            axes3.tick_params(labelsize=6)
            axes4.tick_params(labelsize=6)
            
            
            # ピーク検出のプロット
            axes[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].plot(data[0], label = 'ppg', linewidth=0.5)
            axes[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].plot(peak_times, peak_vals, marker = 'o', markersize=0.1, label = 'peak', linewidth=0.5)
            axes[idx//6*da+int((shift+args.shift_num*args.shift_width)/args.shift_width)][idx%6].tick_params(labelsize=6)
            #plt.legend()
        
    #fig2.savefig(args.dir_out+"1st_split_data_plot.png")
    #fig.savefig(args.dir_out+"peak_detect.png")
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--fs", type=int, default=125)
    parser.add_argument("--pre_sec", type=float, default=0.5)
    parser.add_argument("--post_sec", type=float, default=0.1)
    parser.add_argument("--subject_num", type=int, default=5)
    parser.add_argument("--shift_width", type=int, default=1)
    parser.add_argument("--shift_num", type=int, default=4)

    parser.add_argument("--dir_in", default='2_FilteredData/')
    parser.add_argument("--dir_out", default='3_SplitData/')
    parser.add_argument("--filelist_dir", default='0_FileList/')
    parser.add_argument("--flistname_in", default='2_Filtered.txt')
    parser.add_argument("--flistname_out", default='3_Split.txt')
    
    args = parser.parse_args()
    main(args)
