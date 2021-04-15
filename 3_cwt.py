# -*- coding: utf-8 -*-
import pycwt as wavelet

import numpy as np
import math
import matplotlib.pyplot as plt
import codecs
import os
import re
from tqdm import tqdm
import argparse
import inspect

def plot():
    mother = wavelet.Morlet(6)
    print(inspect.getsource(wavelet.Morlet))

def main(args):
# ----- パラメータ設定
    pre_sec = args.pre_sec
    post_sec = args.post_sec
    fs = args.fs
    N = int(pre_sec*fs)+int(post_sec*fs)#データ長
    dt = 1/fs
    t = np.arange(0,N)*dt # time array
    mother = wavelet.Morlet(6)
    s0 = 2*dt # ウェーブレットの最小スケール。デフォルト値2dt。
    dj = 1/12 # 離散スケールの間隔。デフォルト値1/12。
    J =(math.log2(N * dt / s0))/dj # スケールの範囲s0からs0*2**(J*dj)までで、計(J+1)の尺度。
    print(J)

# ----- データ入出力するディレクトリ
    dir_in = args.dir_in
    dir_out = args.dir_out
    filelist_dir = args.filelist_dir
    flistname_in = args.flistname_in
    flistname_out = args.flistname_out

    if not os.path.exists(filelist_dir):
        os.mkdir(filelist_dir)
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    #データセットをファイルリストとしてテキストファイルに書き込みたい。
    filelist_out = filelist_dir + flistname_out
    with open(filelist_out, mode='w') as _f:
        pass #ここではファイル作るだけ

    #ファイルリスト読み込み
    filelist = filelist_dir + flistname_in
    filelist_fp = codecs.open(filelist, 'r')

# ----- ファイル読み込み
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        split_data = np.load(fname)
        npeaks = split_data.shape[0]
        wavelet_data = np.empty(0)

    # ----- iq差分の2段階増幅だけcwt。split_data[i]の中には順にppg,i1,i2,q1,q2,iqdiff1,iqdiff2が入っている。
        for peak_idx in range(npeaks):
            iq_diff_2nd = split_data[peak_idx][6]
            wave, _scales, _freqs, _coi, _fft, _fftfreqs = wavelet.cwt(iq_diff_2nd, dt, dj, s0, J, mother)
            wavelet_data = np.append(wavelet_data,wave)
            #iwave_iq_diff_2nd = wavelet.icwt(wave_iq_diff_2nd, scales_iq_diff_2nd, dt, dj, mother) #icwt

    # ----- npy形式で保存
        wavelet_data = np.reshape(wavelet_data, (npeaks,math.ceil(J)+1,N))
        print(wavelet_data.shape)
        filename_out = fname.replace(dir_in, dir_out)
        np.save(filename_out, wavelet_data)
    # ----- ファイルリストにファイル名保存
        with open(filelist_out, mode='a') as f:
            f.write(filename_out + '\n')

        """12.11変更　iqdiff2ndのみ絶対値保存"""
        #data_out = np.abs(wave_iq_diff_2nd)
        #np.savetxt(dir_out+'/'+filename_out, data_out, delimiter=',')

    # 元波形プロット
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, iq_diff_2nd, color='c')

    ax.set_xlim(0, t[-1])

    """
    # 逆wavelet波形プロット
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(t, iwave, color='blue')
    ax2 = ax.twinx()
    ax2.plot(t, iwave_ppg, color='orange')
    ax.set_xlim(0, t[-1])"""


    """
    # スペクトログラムのプロット
    plt.figure()
    plt.pcolormesh(t, freqs, np.abs(wave), vmin=0)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim([10, 40])
    #plt.yscale('log')
    """
    """
    fig2=plt.figure()
    ax=fig2.add_subplot(1,1,1)
    ax.pcolormesh(t, freqs_q_2nd, np.abs(wave_q_2nd), vmin=0)
    ax.set_ylabel('Q_2nd_Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_ylim([3, 50])"""
    """
    ax=fig2.add_subplot(1,2,2)
    ax.pcolormesh(t, freqs_ppg, np.abs(wave_ppg), vmin=0)
    ax.set_ylabel('PPG_Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_ylim([1.6, 10])"""

    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--fs", type=int, default=125)
    parser.add_argument("--pre_sec", type=float, default=0.5)
    parser.add_argument("--post_sec", type=float, default=0.1)
    parser.add_argument("--subject_num", type=int, default=5)

    parser.add_argument("--dir_in", default='3_SplitData/')
    parser.add_argument("--dir_out", default='4_WaveletData/')
    parser.add_argument("--filelist_dir", default='0_FileList/')
    parser.add_argument("--flistname_in", default='3_Split.txt')
    parser.add_argument("--flistname_out", default='4_Wavelet.txt')

    args = parser.parse_args()
    main(args)
    #plot()
