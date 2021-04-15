import numpy as np
import codecs
import re
from tqdm import tqdm

def get_y_dim(filelist, disp=False):
    """
    被験者数を取得
    """
    subjects = {}
    y_dim = 0

    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        if subject not in subjects:
            subjects[subject] = y_dim
            y_dim = y_dim + 1

    if disp:
        print(f"y_dim = {y_dim}")

    return y_dim

def count_peaks(filelist, disp=False):
    """
    それぞれの被験者のデータが何ピークあるか取得
    """
    peaks = {}
    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        data = np.load(fname)
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        if subject not in peaks:
            peaks[subject] = 1
        else:
            peaks[subject] = peaks[subject] + 1
    
    if disp:
        print("########### 被験者ごとのピーク数(全体) ##########")
        for subject, numofpeaks in peaks.items():
            print(f"被験者：{subject},  ピーク数：{numofpeaks}")

    return peaks

def count_peaks_distance(filelist, distance, disp=False):
    """
    それぞれの距離で、被験者のデータが何ピークあるか取得
    """
    peaks = {}
    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        data_distance = re.findall('[A-Z][a-z]*_(.*)_rec',fname)[0] #距離　ex)50cm
        if data_distance == distance:
            if subject not in peaks:
                peaks[subject] = 1
            else:
                peaks[subject] = peaks[subject] + 1
    
    if disp:
        print(f"########### 被験者ごとのピーク数(距離:{distance}) ##########")
        for subject, numofpeaks in peaks.items():
            print(f"被験者：{subject},  ピーク数：{numofpeaks}")

    return peaks

def count_peaks_breathing(filelist, with_breathing=True, disp=False):
    """
    それぞれの被験者のデータが何ピークあるか取得
    """
    peaks = {}

    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        if with_breathing: 
            if "wo_breath" not in fname:
                if subject not in peaks:
                    peaks[subject] = 1
                else:
                    peaks[subject] = peaks[subject] + 1
        else:
            if "wo_breath" in fname:
                if subject not in peaks:
                    peaks[subject] = 1
                else:
                    peaks[subject] = peaks[subject] + 1
    
    if disp:
        whether = "w/ breathing" if with_breathing else "wo/ breathing"
        print(whether)
        for subject, numofpeaks in peaks.items():
            print(f"被験者：{subject},  ピーク数：{numofpeaks}")

    return peaks

def count_peaks_pre250(filelist, disp=False):
    """
    それぞれの被験者のデータが何ピークあるか取得
    """
    peaks = {}
    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        data_pre = int(re.findall('pre_([0-9]+)_post',fname)[0]) #ピーク前何サンプルとってるか
        if data_pre == 250:
            if subject not in peaks:
                peaks[subject] = 1
            else:
                peaks[subject] = peaks[subject] + 1
    
    if disp:
        print("########### 被験者ごとのピーク数(pre250,post50) ##########")
        for subject, numofpeaks in peaks.items():
            print(f"被験者：{subject},  ピーク数：{numofpeaks}")

    return peaks

def count_peaks_distance_pre250(filelist, distance, disp=False):
    """
    それぞれの距離で、被験者のデータが何ピークあるか取得
    """
    peaks = {}
    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        data_distance = re.findall('[A-Z][a-z]*_(.*)_rec',fname)[0] #距離　ex)50cm
        data_pre = int(re.findall('pre_([0-9]+)_post',fname)[0]) #ピーク前何サンプルとってるか
        if data_distance == distance and data_pre == 250:
            if subject not in peaks:
                peaks[subject] = 1
            else:
                peaks[subject] = peaks[subject] + 1
    if disp:
        print(f"########### 被験者ごとのピーク数(距離:{distance}) ##########")
        for subject, numofpeaks in peaks.items():
            print(f"被験者：{subject},  ピーク数：{numofpeaks}")

    return peaks

def count_peaks_detail(filelist, disp=False):
    """
    各被験者、各距離、各分割(pre,post)でそれぞれ何ピークあるかカウント
    その中で最小のピーク数も返す
    """
    peaks = {} # 被験者、距離、prepostの３ネスト構造の辞書を想定

    # ファイルリスト読み込み
    filelist_fp = codecs.open(filelist, 'r')
    for _idx, filename in tqdm(enumerate(filelist_fp)):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        fname = split_fpdata[0]
        data = np.load(fname)
        npeaks = data.shape[0]
        # fname例1)4_WaveletSplitData/0_filter_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        # fname例2)4_WaveletSplitData/0_filter_wo_breath_20201012_151747_Manabe_50cm_rec_front_pre_250_post_50_0.csv
        subject = re.findall('_([A-Z][a-z]*)_',fname)[0] #被験者 ex)Manabe
        data_distance = re.findall('[A-Z][a-z]*_(.*)_rec',fname)[0] #距離　ex)50cm
        data_pre = int(re.findall('pre_([0-9]+)_post',fname)[0]) #ピーク前何サンプルとってるか
        is_with_breathing = True if "wo_breath" not in fname else False #呼吸ありか無しか
        if is_with_breathing:
            if subject not in peaks:
                # 被験者がまだ登録されていなければ、登録
                peaks[subject] = {data_distance:{data_pre:npeaks}}
            elif data_distance not in peaks[subject]:
                # 当該被験者で、距離がまだ登録されていなければ登録
                peaks[subject][data_distance] = {data_pre:npeaks}
            elif data_pre not in peaks[subject][data_distance]:
                # 当該被験者の当該距離で、分割（pre,post）がまだ登録されていなければ登録
                peaks[subject][data_distance][data_pre] = npeaks

    if disp:
        print(f"########### 被験者ごとのピーク数(詳細) ##########")
        for subject, peaks_sub in peaks.items():
            print(f"[被験者：{subject}]")
            for dist, peaks_sub_dist in peaks_sub.items():
                print(f" *　距離：{dist}")
                for pre, numofpeaks in peaks_sub_dist.items():
                    #print(f"被験者：{subject},  距離：{dist},　Pre：{pre}, ピーク数：{numofpeaks}")
                    print(f"  ・  Pre：{pre}, ピーク数：{numofpeaks}")
            print()

    min_peaks = 99999
    for subject, peaks_sub in peaks.items():
        for dist, peaks_sub_dist in peaks_sub.items():
            for pre, numofpeaks in peaks_sub_dist.items():
                min_peaks = min(min_peaks, numofpeaks)
    print(min_peaks)

    return peaks, min_peaks

if __name__ == '__main__':
    count_peaks_detail('0_FileList/4B_Sma5Wavelet.txt',disp=True)