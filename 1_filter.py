"""
2021.01.18版
./0_FileList/1_Raw.txtに書かれた計測データを指定されたインデックスの分だけフィルタリング
fs=500
"""
from scipy import signal
import codecs
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def hpf(d_in, sampling_rate, fp, fs):
    """ high pass filter """
    # fp = 0.5   # 通過域端周波数[Hz]→入力引数に変更
    # fs = 0.1   # 阻止域端周波数[Hz]→入力引数に変更
    gpass = 1   # 通過域最大損失量[dB]
    gstop = 40  # 阻止域最小減衰量[dB]

    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(wp=norm_pass, ws=norm_stop,
                            gpass=gpass, gstop=gstop, analog=0)
    b, a = signal.cheby2(N, gstop, Wn, "high")
    d_out = signal.filtfilt(b, a, d_in)
    return d_out

def lpf(d_in, sampling_rate, fp, fs):
    """ low pass filter """
    # fp = 30    # 通過域端周波数[Hz]→入力引数に変更
    # fs = 50    # 阻止域端周波数[Hz]→入力引数に変更
    gpass = 1   # 通過域最大損失量[dB]
    gstop = 40  # 阻止域最小減衰量[dB]

    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(wp=norm_pass, ws=norm_stop,
                            gpass=gpass, gstop=gstop, analog=0)
    b, a = signal.cheby2(N, gstop, Wn, "low")
    d_out = signal.filtfilt(b, a, d_in)
    return d_out

def main(args):
    filelist_dir = args.filelist_dir
    if not os.path.exists(filelist_dir):
        os.mkdir(filelist_dir)
    dirout = args.dir_out
    if not os.path.exists(dirout):
        os.mkdir(dirout)

    # ファイルリスト読み込み
    filelist = filelist_dir+args.flistname_in
    filelist_fp = codecs.open(filelist, 'r')

    #ファイルリストとしてテキストファイルに書き込みたい。
    filelist_out = filelist_dir+args.flistname_out
    with open(filelist_out, mode='w') as f:
        pass #何もしない。データ作るたびにそのcsvファイル名を追記。
    
    fig = plt.figure()
    fig_before = plt.figure()

    for idx, filename in enumerate(filelist_fp):
        # ファイル読み込み
        split_fpdata = filename.rstrip('\r\n').split(',')
        wave_fp = codecs.open(split_fpdata[0], 'r')
        print(
            f"Input file: {split_fpdata[0]}, {split_fpdata[1]}, {split_fpdata[2]}")
        times = []
        ch = []
        for i in range(7):
            ch.append([])
        for idx2, line in enumerate(wave_fp):
            split_data = line.rstrip('\r\n').split(',')
            times.append(idx2)
            for i in range(7):
                ch[i].append(int(float(split_data[i])))
        wave_fp.close()

        ppg = lpf(ch[0], 500, fp=args.ppg_lpf_fp, fs=args.ppg_lpf_fs)[::args.thinning_num]  # PPG
        i_1st = ch[1]
        i_2nd = ch[2]
        q_1st = ch[3]
        q_2nd = ch[4]
        iq_diff_1st = [i-q for (i,q) in zip(i_1st, q_1st)]
        iq_diff_2nd = [i-q for (i,q) in zip(i_2nd, q_2nd)]
        
        ##########  filter前プロット部分  ##########
        iq_diff_2nd_b = iq_diff_2nd[int(split_fpdata[1]):int(split_fpdata[1])+30000]
        iq_diff_2nd_wob = iq_diff_2nd[int(split_fpdata[2])-5000:int(split_fpdata[2])]
        #print(f"呼吸あり：{len(iq_diff_2nd_b)}, 呼吸なし：{len(iq_diff_2nd_wob)}")
        ax = fig_before.add_subplot(args.subject_num,3,idx+1)
        ax.plot(iq_diff_2nd_b,label='w/ breathing')
        ax.plot(iq_diff_2nd_wob,label='w/o breathing')
        ax.legend()

        #フィルターをかける
        i_1st = hpf(lpf(i_1st, 500, fp=args.doppler_lpf_fp, fs=args.doppler_lpf_fs), 500, fp=args.doppler_hpf_fp, fs=args.doppler_hpf_fs)[::args.thinning_num] # 4sampleずつ間引く
        i_2nd = hpf(lpf(i_2nd, 500, fp=args.doppler_lpf_fp, fs=args.doppler_lpf_fs), 500, fp=args.doppler_hpf_fp, fs=args.doppler_hpf_fs)[::args.thinning_num]
        q_1st = hpf(lpf(q_1st, 500, fp=args.doppler_lpf_fp, fs=args.doppler_lpf_fs), 500, fp=args.doppler_hpf_fp, fs=args.doppler_hpf_fs)[::args.thinning_num]
        q_2nd = hpf(lpf(q_2nd, 500, fp=args.doppler_lpf_fp, fs=args.doppler_lpf_fs), 500, fp=args.doppler_hpf_fp, fs=args.doppler_hpf_fs)[::args.thinning_num]
        iq_diff_1st = hpf(lpf(iq_diff_1st, 500, fp=args.doppler_lpf_fp, fs=args.doppler_lpf_fs), 500, fp=args.doppler_hpf_fp, fs=args.doppler_hpf_fs)[::args.thinning_num]
        iq_diff_2nd = hpf(lpf(iq_diff_2nd, 500, fp=args.doppler_lpf_fp, fs=args.doppler_lpf_fs), 500, fp=args.doppler_hpf_fp, fs=args.doppler_hpf_fs)[::args.thinning_num]

        # 出力先ファイル
        #例）./filter2/filter_20201012_151747_Manabe_50cm_rec_front.csv
        filename_out = dirout+str(idx)+'_filter_'+split_fpdata[0].replace('./','').replace(args.dir_in,'').replace('csv','npy')
        filename_out_wo_breath = dirout+str(idx)+'_filter_wo_breath_'+split_fpdata[0].replace('./','').replace(args.dir_in,'').replace('csv','npy')

        ##########　通常呼吸　##########
        X_LIM_MIN = int(split_fpdata[1])/args.thinning_num
        X_LIM_MIN = int(X_LIM_MIN)
        X_LIM_MAX = X_LIM_MIN+30000/args.thinning_num #そこから60sは通常の呼吸
        X_LIM_MAX = int(X_LIM_MAX)

        print(f"X_LIM_MIN={X_LIM_MIN}, X_LIM_MAX={X_LIM_MAX}")
        chnp = []
        chnp.append(np.array(ppg)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(i_1st)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(i_2nd)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(q_1st)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(q_2nd)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(iq_diff_1st)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(iq_diff_2nd)[X_LIM_MIN:X_LIM_MAX])
        dout = np.vstack([chnp[0], chnp[1], chnp[2], chnp[3], chnp[4], chnp[5], chnp[6]])

        #np.savetxt(filename_out, np.vstack(
        #    [chnp[0], chnp[1], chnp[2], chnp[3], chnp[4], chnp[5], chnp[6]]), delimiter=',')
        np.save(filename_out,dout)

        ax = fig.add_subplot(args.subject_num,3,idx+1)
        ax.plot(chnp[6], label='w/ breathing')

        # ファイルリストにファイル名保存
        with open(filelist_out, mode='a') as f:
            #f.write('2_FilteredData/'+filename_out.replace(dirout+'/','') + '\n')
            f.write(filename_out+'\n')
        
        ########## 呼吸なし ##########
        if int(split_fpdata[2]) == -1:
            X_LIM_MAX = -1
        else:
            X_LIM_MAX = int(split_fpdata[2])/args.thinning_num
            X_LIM_MAX = int(X_LIM_MAX)
        X_LIM_MIN = X_LIM_MAX - 5000/args.thinning_num
        X_LIM_MIN = int(X_LIM_MIN)

        print(f"X_LIM_MIN={X_LIM_MIN}, X_LIM_MAX={X_LIM_MAX}")
        chnp = []
        chnp.append(np.array(ppg)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(i_1st)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(i_2nd)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(q_1st)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(q_2nd)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(iq_diff_1st)[X_LIM_MIN:X_LIM_MAX])
        chnp.append(np.array(iq_diff_2nd)[X_LIM_MIN:X_LIM_MAX])
        dout = np.vstack([chnp[0], chnp[1], chnp[2], chnp[3], chnp[4], chnp[5], chnp[6]])
        #np.savetxt(filename_out_wo_breath, np.vstack(
        #    [chnp[0], chnp[1], chnp[2], chnp[3], chnp[4], chnp[5], chnp[6]]), delimiter=',')
        np.save(filename_out_wo_breath,dout)

        # ファイルリストにファイル名保存
        with open(filelist_out, mode='a') as f:
            #f.write('2_FilteredData/'+filename_out_wo_breath.replace(dirout+'/','') + '\n')
            f.write(filename_out_wo_breath + '\n')

        ax.plot(chnp[6],label='w/o breathing')
        ax.legend()
    fig_before.savefig(dirout+'/before_filter.png')
    fig.savefig(dirout+'/filter.png')
    #plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppg_lpf_fp", type=int, default=5)
    parser.add_argument("--ppg_lpf_fs", type=int, default=10)
    parser.add_argument("--doppler_lpf_fp", type=float, default=40)
    parser.add_argument("--doppler_lpf_fs", type=float, default=50)
    parser.add_argument("--doppler_hpf_fp", type=float, default=3)
    parser.add_argument("--doppler_hpf_fs", type=float, default=0.4)
    parser.add_argument("--subject_num", type=int, default=5)

    parser.add_argument("--dir_in", default='1_RawData/')
    parser.add_argument("--dir_out", default='2_FilteredData/')
    parser.add_argument("--filelist_dir", default='0_FileList/')
    parser.add_argument("--flistname_in", default='1_Raw.txt')
    parser.add_argument("--flistname_out", default='2_Filtered.txt')
    parser.add_argument("--thinning_num", type=int, default=4)
    args = parser.parse_args()
    main(args)

"""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=256)#128
    parser.add_argument("--val_batch_size", type=int, default=256)#128
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--datalength", type=int, default=300)
    parser.add_argument("--enc_convlayer_sizes", type=list,
                        default=[[1, 1], [4, 2], [4, 2]])
    parser.add_argument("--enc_fclayer_sizes", type=list,
                        default=[300, 128, 64])
    parser.add_argument("--dec_fclayer_sizes", type=list,
                        default=[64, 128, 300])
    parser.add_argument("--dec_convlayer_sizes", type=list, default=[])
    # loss 280 -> 170
    # parser.add_argument("--datalength", type=int, default=400)
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[1, 1], [4, 2], [4, 2]])
    # parser.add_argument("--enc_fclayer_sizes", type=list, default=[400, 128, 64])
    # parser.add_argument("--dec_fclayer_sizes", type=list, default=[64, 128, 400])
    # parser.add_argument("--dec_convlayer_sizes", type=list, default=[])
    # loss 240~250 -> 160 -> 120~130
    # # parser.add_argument("--datalength", type=int, default=300)
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[1, 1], [2, 2], [2, 2]])
    # parser.add_argument("--enc_fclayer_sizes", type=list, default=[150, 128, 64])
    # parser.add_argument("--dec_fclayer_sizes", type=list, default=[64, 128, 150])
    # parser.add_argument("--dec_convlayer_sizes", type=list, default=[[2, 2], [2, 2], [1, 1]])
    parser.add_argument("--latent_size", type=int, default=20)#5
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--train_off", action='store_false')"""