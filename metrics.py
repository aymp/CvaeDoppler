"""
csv_output.pyで出力した
6_ClusteringResults/hoge/cvae_output.csv
から、FARとFRR / FPIRとFNIRを算出したい
"""

from re import T
import numpy as np
import pandas as pd
import argparse

import matplotlib.pyplot as plt

def main(args):
# ----- パラメータ設定
    DISTANCE = args.distance
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
        CHILD_DIR_OUT = 'sma'+str(SMA_NUM)+'_zdim'+str(Z_DIM)+'_pool'+str(POOL)+'_'+DISTANCE+'_epoch'+str(EPOCH_NUM)+'_ydim'+str(Y_DIM)+'/'

    DIR_OUT = PARENT_DIR_OUT + CHILD_DIR_OUT
    
# ----- CVAEの出力データを保存してるCSVファイルを読み込む
    cvae_output_file_path = DIR_OUT+'cvae_output.csv'
    df = pd.read_csv(cvae_output_file_path, header=None, names=('true_label','pred_label','z1','z2','z3','index','trained_flag'))

# ----- 訓練済みデータの線税変数の中心座標を計算する

    z_trained_avg = {} # 中心座標を人物ごとに保存するdict
    trained_df = df.query('trained_flag') # trained_flag が True の行を抽出する

    # モデルの学習に用いたデータのうち、被験者番号早い順に見て行って潜在変数の平均出す
    for tl in range(Y_DIM):
        df_temp = trained_df.query('true_label == @tl')[['z1','z2','z3']]
        z_trained_avg[tl] = df_temp.mean()
    
# ----- 閾値を変えながら指標計算する
    metrics = {}
    metrics['th_val'] = []
    metrics['fnir'] = []
    metrics['fpir'] = []

    for TH_VAL in [i/100 for i in range(50, 150, 1)]:
        metrics['th_val'].append(TH_VAL)
    ################################################################################
    #                FNIR(登録済み人物が認証に失敗する割合)を計算する                 #
    ################################################################################

    # ----- STEP1：true_label < Y_DIM の個数が分母となる(登録済みの人物の心拍での検索件数)
        fnir_df = df.query('true_label < @Y_DIM')
        fnir_d = len(fnir_df) # denominator(分母)

    # ----- STEP2：fnir_dfのうち、予測ラベルが間違えているものをカウントする(分類器ミス)
        fnir_n = 0 # numerator(分子)：別人あるいは未登録人物だと識別される件数

        fnir_classifier_df = fnir_df.query('true_label != pred_label')
        fnir_n += len(fnir_classifier_df)

    # ----- STEP3：fnir_dfのうち、閾値条件をクリアしていないものをカウントする
        fnir_threshold_df = fnir_df.query('true_label == pred_label')

        for tl in range(Y_DIM):

            # 分類器を突破したデータの潜在変数を抽出する
            fnir_threshold_temp_df = fnir_threshold_df.query('true_label == @tl')[['z1','z2','z3']]
            
            # 各被験者の中心座標とのベクトルを計算する
            fnir_threshold_temp_df -= z_trained_avg[tl]

            # ベクトルのノルムを計算する(＝中心座標とプロットの距離)
            fnir_threshold_temp_df = fnir_threshold_temp_df.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
            
            # 閾値を超えているものの個数をFNIRの分子に加算していく
            fnir_n += (fnir_threshold_temp_df > TH_VAL).sum()

    # ----- STEP4：FNIR計算完了
        fnir = fnir_n/fnir_d
        metrics['fnir'].append(fnir)

    ################################################################################
    #           FPIR(登録していない人物が間違って識別される割合)を計算する            #
    ################################################################################

    # ----- STEP1：true_label >= Y_DIM の個数を分母とする
        fpir_df = df.query('true_label >= @Y_DIM')
        fpir_d = len(fpir_df)

    # ----- STEP2：fpir_dfのうち、閾値条件をクリアしてしまったものをカウントする
        fpir_n = 0 # numerator(分子)

        for pl in range(Y_DIM):

            # データの潜在変数を抽出する
            fpir_temp_df = fpir_df.query('pred_label == @pl')[['z1','z2','z3']]

            # 各被験者の中心座標とのベクトルを計算する
            fpir_temp_df -= z_trained_avg[pl]

            # ベクトルのノルムを計算する(＝中心座標とプロットの距離)
            fpir_temp_df = fpir_temp_df.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
            
            # 閾値を超えていない（＝突破した）個数をFPIRの分子に加算していく
            fpir_n += (fpir_temp_df <= TH_VAL).sum()

    # ----- STEP3：FPIR計算完了
        fpir = fpir_n/fpir_d
        metrics['fpir'].append(fpir)

        print(TH_VAL, fnir, fpir)

# ----- FNIRとFPIRをグラフにまとめる
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = metrics['th_val']
    y = metrics['fnir']
    ax.plot(x, y, label='fnir')
    y = metrics['fpir']
    ax.plot(x, y, label='fpir')
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--distance", default='75cm')
    parser.add_argument("--y_dim", type=int, default=5)
    parser.add_argument("--z_dim", type=int, default=3)
    parser.add_argument("--epoch_num", type=int, default=100)
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