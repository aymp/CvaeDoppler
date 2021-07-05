# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:52:03 2019

信号処理用の関数ライブラリ


@author: cs 28
"""
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

################################################################################
#        そのうち移動させる
################################################################################
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data2 - data1                   # Difference between data1 and data2
    #thre      = np.abs(diff)<90
    diff      = diff
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    #plt.figure()
    plt.figure(figsize=(6,4),dpi=100)
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--', linewidth=2)
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=2)
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=2)
    plt.ylim(-150,150)
    print('Mean of difference: '+ str(md) +'ms')
    print('Standard deviation of the difference: '+ str(sd) +'ms')

################################################################################
#                                 移動平均                                      #
################################################################################

# Simple Moving Average Filter
#基線変動除去に用いるときはpoints = fs
def sma( signal, points ):
    mva=np.ones(points)/float(points) #移動平均用
    return np.convolve(np.concatenate([signal[:points],signal,signal[-points:]]), mva, mode='same')[points:-points]

# Linear Weighted Moving Average
def lwma( signal, points):
    w = (np.arange(points)+1)/(points*(points+1)/2)
    ex_sig = np.concatenate([signal[:points],signal,signal[-points:]])
    return np.convolve(ex_sig, w, mode='same')[points:-points]

# Exponential Moving Average
@jit(nopython=True)
def ema( signal, alpha ):
    if not(0 < alpha < 1):
        print("The 3rd value is not in 0-1!")
        return signal
    
    y = np.empty_like(signal)
    y[0] = signal[0]
    for i in range(1,len(signal)):
        y[i] = alpha*signal[i] + (1-alpha)*y[i-1]
    return y

# Running Moving Average
def rma( signal, points ):
    return ema(signal, 1.0/points)

def sma_filtfilt( signal, points ):
    front = sma(signal, points)
    back = np.array( list(reversed(sma(list(reversed(signal)), points))) )
    return (front + back)/2
    
################################################################################
################################################################################




################################################################################
#                                 ピーク検出                                     #
################################################################################
# peak detection with zero cross
""" diffして0クロス点からピーク抽出
    閾値判別により誤検出の除去
    極大点検出機能の追加
    output：pos peak """
def pd0cross(mva_ppg, fs, *, ADJ_LOOP=2, ADJ_COEF=0.35, th_on=True ):
    dif = np.diff(mva_ppg)*10
    tmp_pk_list =np.where(np.diff(np.sign(dif)))[0]+1 #全てのピークを取得
    pos_pk_list, neg_pk_list = [], []   #正のピーク及び負のピーク
    #THRESHOLD = sorted(mva_ppg[tmp_pk_list])[-5]*0.45 if th_on else 0  #だいたい10番目くらいの値を基準に.
    THRESHOLD = sorted(dif)[-5*int(fs/16)]*0.6 if th_on else 0 #だいたい10番目くらいの値を基準に.
    #print(THRESHOLD)
    DATA_LEN = len(mva_ppg)  #境界判別
    #ADJ_LOOP = 0             #ピーク位置調整の回数
    ADJ_POINT = int(fs*ADJ_COEF) #調整点数
    
    for i in range(1,len(tmp_pk_list)):
        #有効なピークであるか閾値を用いて判別
        if np.any(np.where(dif[tmp_pk_list[i-1]:tmp_pk_list[i]+1] > THRESHOLD)):
            pos_pk_list.append(tmp_pk_list[i])
        if dif[tmp_pk_list[i-1]-1] <= dif[tmp_pk_list[i-1]]:
            #負のピークはすべて検出する
            neg_pk_list.append(tmp_pk_list[i-1])
    if dif[tmp_pk_list[i]-1] <= dif[tmp_pk_list[i]]:
        neg_pk_list.append(tmp_pk_list[i])
    
    pos_pk_list = np.array(pos_pk_list)
    
    #有効な正のピークは両隣に負のピークが存在するはず
    pk_list = np.array(pos_pk_list[np.where( (neg_pk_list[0]<pos_pk_list) & (pos_pk_list<neg_pk_list[-1]) )])
    
    for loop in range(ADJ_LOOP):
        #ループ回数分ピーク位置微調整を繰り返す
        for idx, pk_idx in enumerate(pk_list):
            #ピーク位置の微調整
            l_border, adj = (pk_idx-ADJ_POINT, ADJ_POINT) if (pk_idx-ADJ_POINT) > 0 else (0, pk_idx)
            r_border = pk_idx+ADJ_POINT if (pk_idx+ADJ_POINT) < DATA_LEN-1 else DATA_LEN-1
            pk_list[idx] = pk_idx + ( np.argmax(mva_ppg[l_border : r_border+1])-adj)
        pk_list = np.unique(pk_list)

    return pk_list


#時系列探索
def peak_search(data_frame, sampling_rate):
    """ peak search for ecg """
    peak_times = []
    peak_vals = []
    temp_max = [-1, -9999]
    temp_min = [-1, 9999]
    max_search_flag = True
    max_ratio = 0.4
    shift_rate = sampling_rate
    
    shift_min = int(0.45 * sampling_rate)
    shift_max = int(0.8 * sampling_rate)
    first_skip = int(0.1 * shift_rate)
    finish_search = int(0.45* shift_rate)
    for idx, val in enumerate(data_frame):
        if (idx < first_skip):
            continue
        if (max_search_flag or (idx - temp_max[0] > shift_min)) and val >= temp_max[1] or (idx - temp_max[0] > shift_max):
            temp_max = [idx, val]
            max_search_flag = True
        if val < temp_min[1]:
            temp_min = [idx, val]
        if max_search_flag and (idx - temp_max[0] > finish_search):
            peak_times.append(temp_max[0])
            peak_vals.append(temp_max[1])
            temp_max[1] -= (temp_max[1] - temp_min[1]) * (1.0 - max_ratio)
            temp_min = [None, 999]
            max_search_flag = False
    return peak_times, peak_vals


################################################################################
################################################################################

################################################################################
#                                誤差補正                                      #
################################################################################
""""
相互相関 zncc
signal: 補正する信号
pt_sig: 信号のピーク位置
l,r,s : 順に 左ウィンドウ幅, 右〃, 探索範囲
"""
def error_compensate(signal, pt_sig, l, r, s, *, REF_SR=1024, PLOT_FLAG=False,TITLE=''):
    #REF_SR = 1024
    #points of a window
    win_l = int(REF_SR * l)
    win_r = int(REF_SR * r)
    win_s = int(REF_SR * s)
    #output vals
    ppi_list = []
    shift_list = []
    pop_list = []
    
    break_flag = False
    
    for idx in range(0, len(pt_sig) - 1):
        corr_score = np.zeros(win_s*2+1)
        
        tw_border_l = int(pt_sig[idx] - win_l)
        tw_border_r = int(pt_sig[idx] + win_r)
        if tw_border_l < 0:
            pop_list.append(idx)
            continue
        else:
            tw = signal[tw_border_l:tw_border_r]
            tw = tw - np.mean(tw)
        
        for dt in range(-win_s, win_s+1):
            corr_idx = dt + win_s
            sw_border_r = int(pt_sig[idx + 1] + win_r + dt) #サーチウィンドウ境界
            sw_border_l = int(pt_sig[idx + 1] - win_l + dt)
            if sw_border_r >= len(signal):
                pop_list.append(idx+1)
                break_flag = True
                break
            sw = signal[sw_border_l:sw_border_r]
            sw = sw - np.mean(sw)
            conv = np.dot(tw, sw)
            dev = np.sqrt(np.sum(tw**2)) * np.sqrt(np.sum(sw**2))
            if dev == 0: corr_score[corr_idx] = 0
            else: corr_score[corr_idx] = conv / dev
        if break_flag:
            continue
        
        #print(max(corr_score)) if max(corr_score) < 0.8 else max(corr_score)
        pt_corr = corr_score.argmax() - win_s
        ppi_list.append(pt_sig[idx+1] - pt_sig[idx] + pt_corr)
        #pt_sig[idx+1] += pt_corr
        shift_list.append(pt_corr)
        
        if PLOT_FLAG:
            plt.figure()
            plt.subplot(3,1,1)
            plt.title(TITLE+':'+str(idx))
            plt.plot(np.arange(win_s,win_s+len(tw)), tw/dev)
            plt.scatter(np.arange(win_s,win_s+len(tw))[win_l], tw[win_l]/dev, color='purple',s=18)
            tmp=(signal[int(pt_sig[idx + 1] - win_l -win_s):int(pt_sig[idx + 1] + win_r + win_s)]-np.mean(signal[int(pt_sig[idx + 1] - win_l -win_s):int(pt_sig[idx + 1] + win_r + win_s)]))/dev
            plt.plot(tmp)
            plt.scatter(np.arange(0,len(tmp))[win_s + win_l +pt_corr], tmp[win_s + win_l +pt_corr], color='red',s=18)
            plt.subplot(3,1,2)
            plt.plot(np.arange(win_s,win_s+len(tw)), tw/dev)
            plt.plot(np.arange(win_s,win_s+len(tw)),(signal[int(pt_sig[idx + 1] - win_l + pt_corr):int(pt_sig[idx + 1] + win_r + pt_corr)]-np.mean(signal[int(pt_sig[idx + 1] - win_l + pt_corr):int(pt_sig[idx + 1] + win_r + pt_corr)]))/dev)
            plt.subplot(3,1,3)
            plt.scatter(np.arange(win_l,win_l+len(corr_score)),corr_score, s=18)
            plt.scatter(np.arange(win_l,win_l+len(corr_score))[np.argmax(corr_score)], corr_score[np.argmax(corr_score)], color='orange')
            plt.scatter(np.arange(win_l,win_l+len(corr_score))[int(len(corr_score)/2)], corr_score[int(len(corr_score)/2)], color='black')
            plt.ylim(min(corr_score)*0.99,max(corr_score)*1.01)
        
    return np.array(ppi_list), np.array(shift_list), np.delete(pt_sig, pop_list)





#############

# Mean Absolute Error 
# [ np.array([...]), ..., np.array([...]) ]という構造用
def MAE(error_list):
    return list(map( np.mean, list(map( np.abs, error_list)) ))
def tMAE(error_list):
    tmp = np.array(error_list).flatten()
    return np.mean(np.abs(tmp))


# 差分関数の逆(累積和)
# comsum
def inv_diff(dif, f0, *, mva=False, fs=0):
    tmp = dif - np.mean(dif)
    tmp = np.concatenate( ([f0], tmp) )
    return np.cumsum(tmp)


# 引数(value)に最近傍な値を持つarrayのidxを返す
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


"""バグありそう
"""
# 2つのベクトルの共通領域の境界を返す
# 最初と最後の位置だけがずれる可能性があることを前提
# [1, 2.1, 3, 4], [2, 3.3, 3.9, 5, 6] =>  [1, 3], [0, 2]
#
def find_common_vec(vec1, vec2, *, thre=None,offset=0):
    if offset >= 800:
        vec2_ofs = vec2 + 300
    else:
        vec2_ofs = vec2 + offset

    if thre is None:
        ft1, lt1 = find_nearest(vec1, vec2_ofs[0]), find_nearest(vec1, vec2_ofs[-1])
        ft2, lt2 = find_nearest(vec2_ofs, vec1[0]), find_nearest(vec2_ofs, vec1[-1])
    else:
        for idx in range(len(vec2_ofs)):
            ft1 = find_nearest(vec1, vec2_ofs[idx])
            if np.abs(vec1[ft1]-vec2_ofs[idx]) < thre:
                break
        for idx in range(1,len(vec2_ofs)+1):
            lt1 = find_nearest(vec1, vec2_ofs[-idx])
            if np.abs(vec1[lt1]-vec2_ofs[-idx]) < thre:
                break
        for idx in range(len(vec1)):
            ft2 = find_nearest(vec2_ofs, vec1[idx])

            if np.abs(vec2_ofs[ft2]-vec1[idx]) < thre:
                break
        for idx in range(1,len(vec1)+1):            
            lt2 = find_nearest(vec2_ofs, vec1[-idx])
            if np.abs(vec2_ofs[lt2]-vec1[-idx]) < thre:
                break
    return list(range(ft1,lt1+1)), list(range(ft2,lt2+1))

    
# 配列のリストの論理和を取る 配列における複数の例外条件を除去する
# length: 除去したい元配列のlen
# blklist: 除去したい配列インデックス
def get_bool_blklist(length, blklist):
    if (blklist == None) | (blklist == []):
        return np.array([True]*length )
    return np.logical_not(np.any([np.arange(length) == i for i in blklist],axis=0))

# 2021.6.30追加　0-1正規化するだけ
def min_max(x, axis=None):
    """0-1の範囲に正規化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result