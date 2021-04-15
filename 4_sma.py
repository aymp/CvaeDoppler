import numpy as np
import os
import codecs
import re
from tqdm import tqdm
import argparse

import get_info_from_filelist

# cat Sma5Wavelet2Vae.txt | sort -R > Sma5Wavelet2VaeShuffled.txt でテキストファイルシャッフル可能

def min_max(x, axis=None):
    """0-1の範囲に正規化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def sma_wavelet(args):
# ----- パラメータ設定
    LOG_MODE = args.log
    WAVELET_HEIGHT = args.wavelet_height
    WAVELET_WIDTH = args.data_len
    SMA_NUM = args.sma_num

    # 移動平均の数だけデータ配列を保持
    data_storage = np.split(np.zeros((WAVELET_HEIGHT * SMA_NUM, WAVELET_WIDTH)), SMA_NUM)

# ----- データ入出力するディレクトリ
    DIR_IN = args.dir_in
    DIR_OUT = args.dir_out

    FILELIST_DIR = args.filelist_dir
    FLISTNAME_IN = args.flistname_in
    FLISTNAME_OUT = args.flistname_out

    FILELIST_IN_PATH = FILELIST_DIR + FLISTNAME_IN
    FILELIST_OUT_PATH = FILELIST_DIR + FLISTNAME_OUT

    if not os.path.exists(FILELIST_DIR):
        os.mkdir(FILELIST_DIR)
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)
    
    # ファイルリスト作成
    with open(FILELIST_OUT_PATH, mode='w') as _f:
        pass #ここではファイル作るだけ

# ----- ファイルリスト読み込み
    filelist_fp = codecs.open(FILELIST_IN_PATH, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
    # ----- 各ファイルからウェーブレット変換データ読み込み。shape:(npeaks, height64, width74)
        split_fpdata = filename.rstrip('\r\n').split(',')
        #print(f"Input file: {split_fpdata[0]}")
        fname = split_fpdata[0]
        wavelet_data = np.load(fname)
        wavelet_data = np.abs(wavelet_data) #←ここ絶対値取るか否かで結果まったく異なる。
        if LOG_MODE:
            wavelet_data = np.log(wavelet_data) # 自然対数とる
        npeaks = wavelet_data.shape[0]
        sma_data_out = np.empty(0)
    # ----- 移動平均
        for peak_idx in range(npeaks):
            data_storage[0] = wavelet_data[peak_idx]
            if peak_idx >= SMA_NUM-1:
                sma_data_temp = data_storage[0]
                for i in range(1,SMA_NUM):
                    sma_data_temp += data_storage[i]
                sma_data_temp /= SMA_NUM
                sma_data_temp = min_max(sma_data_temp)
                sma_data_out = np.append(sma_data_out,sma_data_temp)
        # ----- update
            for i in range(SMA_NUM-1,0,-1):
                data_storage[i] = data_storage[i-1]

    # ----- npy形式で保存
        sma_data_out = np.reshape(sma_data_out, (npeaks-(SMA_NUM-1),WAVELET_HEIGHT,WAVELET_WIDTH))
        data_out_path = fname.replace(DIR_IN, DIR_OUT)
        np.save(data_out_path, sma_data_out)
    # ----- ファイルリストにファイル名保存
        with open(FILELIST_OUT_PATH, mode='a') as f:
            f.write(data_out_path+'\n')

    filelist_fp.close()

def sma_time(args):
# ----- パラメータ設定
    DATA_LEN = args.data_len
    DATA_CH = 7 #ppg~iqdiff2までの7ch
    SMA_NUM = args.sma_num

    # 移動平均の数だけデータ配列を保持
    data_storage = np.split(np.zeros((DATA_CH*SMA_NUM, DATA_LEN)), SMA_NUM)

# ----- データ入出力するディレクトリ
    DIR_IN = args.dir_in
    DIR_OUT = args.dir_out

    FILELIST_DIR = args.filelist_dir
    FLISTNAME_IN = args.flistname_in
    FLISTNAME_OUT = args.flistname_out

    FILELIST_IN_PATH = FILELIST_DIR + FLISTNAME_IN
    FILELIST_OUT_PATH = FILELIST_DIR + FLISTNAME_OUT

    if not os.path.exists(FILELIST_DIR):
        os.mkdir(FILELIST_DIR)
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)
    
    # ファイルリスト作成
    with open(FILELIST_OUT_PATH, mode='w') as _f:
        pass #ここではファイル作るだけ

# ----- ファイルリスト読み込み
    filelist_fp = codecs.open(FILELIST_IN_PATH, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
    # ----- 各ファイルから時系列波形読み込み。shape:(npeaks, ch7, len74)
        split_fpdata = filename.rstrip('\r\n').split(',')
        #print(f"Input file: {split_fpdata[0]}")
        fname = split_fpdata[0]
        time_data = np.load(fname)
        npeaks = time_data.shape[0]
        sma_data_out = np.empty(0)
    # ----- 移動平均
        for peak_idx in range(npeaks):
            data_storage[0] = time_data[peak_idx]
            if peak_idx >= SMA_NUM-1:
                sma_data_temp = data_storage[0]
                for i in range(1,SMA_NUM):
                    sma_data_temp += data_storage[i]
                sma_data_temp /= SMA_NUM
                sma_data_temp = min_max(sma_data_temp)
                sma_data_out = np.append(sma_data_out,sma_data_temp)
        # ----- update
            for i in range(SMA_NUM-1,0,-1):
                data_storage[i] = data_storage[i-1]

    # ----- npy形式で保存
        sma_data_out = np.reshape(sma_data_out, (npeaks-(SMA_NUM-1),DATA_CH,DATA_LEN))
        data_out_path = fname.replace(DIR_IN, DIR_OUT)
        np.save(data_out_path, sma_data_out)
    # ----- ファイルリストにファイル名保存
        with open(FILELIST_OUT_PATH, mode='a') as f:
            f.write(data_out_path+'\n')

    filelist_fp.close()


def main(args):
    if args.time:
        sma_time(args)
    else:
        sma_wavelet(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--time", action='store_true')#デフォFalse
    parser.add_argument("--log", action='store_true')#デフォFalse

    parser.add_argument("--data_len", type=int, default=74)
    parser.add_argument("--wavelet_height", type=int, default=64)
    parser.add_argument("--sma_num", type=int, default=5)

    parser.add_argument("--dir_in", default='4A_WaveletData/')
    parser.add_argument("--dir_out", default='4B_Sma5WaveletData/')
    parser.add_argument("--filelist_dir", default='0_FileList/')
    parser.add_argument("--flistname_in", default='4A_Wavelet.txt')
    parser.add_argument("--flistname_out", default='4B_Sma5Wavelet.txt')
    
    args = parser.parse_args()
    main(args)