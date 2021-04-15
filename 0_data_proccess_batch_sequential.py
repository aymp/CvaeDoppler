""" Data Processing 4 Vae Clustering """
import subprocess

SUBJECT_NUM = '23'
FS = '125'
DATA_LEN = '74'
DATA_HEIGHT = '64'
# WIN_LEN = '32'
# ORDER = '20'
# YULE_LEN = str(int(DATA_LEN)-int(WIN_LEN)+1)
# YULE_HEIGHT = '127'

FILTER_DIR = '2_FilterdData_125Hz/'
FILTER_FILELIST = '2_Filtered_125Hz.txt'

SPLIT_DIR = '3A_SplitData_125Hz/'
SPLIT_FILELIST = '3A_Split_125Hz.txt'

WAVELET_DIR = '4A_WaveletData_125Hz/'
WAVELET_FILELIST = '4A_Wavelet_125Hz.txt'

TIME_SMA_DIR = '3B_Sma5SplitData_125Hz/'
TIME_SMA_FILELIST = '3B_Sma5Split_125Hz.txt'

WAVELET_SMA_DIR = '4B_Sma5WaveletData_125Hz/'
WAVELET_SMA_FILELIST = '4B_Sma5Wavelet_125Hz.txt'

LOG_WAVELET_SMA_DIR = '4C_LogSma5WaveletData_125Hz/'
LOG_WAVELET_SMA_FILELIST = '4C_LogSma5Wavelet_125Hz.txt'

#YULE_DIR = '5A_YuleData_125Hz_w32o20/'
#YULE_FILELIST = '5A_Yule_125Hz_w32o20.txt'

#YULE_SMA_DIR = '5B_Sma5YuleData_125Hz_w32o20/'
#YULE_SMA_FILELIST = '5B_Sma5Yule_125Hz_w32o20.txt'

# 1. FILTER
#subprocess.run(['python',"1_filter.py", "--dir_out",FILTER_DIR,"--flistname_out",FILTER_FILELIST,"--thinning_num",str(4),'--subject_num',SUBJECT_NUM])
# 2. FILTER > SPLIT
#subprocess.run(['python',"2_peak_detect.py","--fs",FS,"--dir_in",FILTER_DIR,"--dir_out", SPLIT_DIR,"--flistname_in",FILTER_FILELIST,"--flistname_out", SPLIT_FILELIST,'--subject_num',SUBJECT_NUM])
# 3. SPLIT > WAVELET
#subprocess.run(['python',"3_cwt.py","--fs",FS,"--dir_in",SPLIT_DIR,"--dir_out",WAVELET_DIR,"--flistname_in",SPLIT_FILELIST,"--flistname_out",WAVELET_FILELIST])
# 4. SPLIT > YULE-WALKER
#subprocess.run(['python3','5_yule_walker.py','--dir_in',SPLIT_DIR,'--dir_out',YULE_DIR,'--flistname_in',SPLIT_FILELIST,'--flistname_out',YULE_FILELIST,'--win_len',WIN_LEN,'--order',ORDER])
# 5-1. SPLIT > SMA
#subprocess.run(['python3',"4_sma.py","--time","--data_len",DATA_LEN,"--dir_in",SPLIT_DIR,"--flistname_in",SPLIT_FILELIST,"--dir_out",TIME_SMA_DIR,"--flistname_out",TIME_SMA_FILELIST])
# 5-2. WAVELET(abs) > SMA
subprocess.run(['python',"4_sma.py","--data_len",DATA_LEN,"--wavelet_height",DATA_HEIGHT,"--dir_in",WAVELET_DIR,"--flistname_in",WAVELET_FILELIST,"--dir_out",WAVELET_SMA_DIR,"--flistname_out",WAVELET_SMA_FILELIST])
# 5-3. WAVELET(log) > SMA
subprocess.run(['python',"4_sma.py","--log","--data_len",DATA_LEN,"--wavelet_height",DATA_HEIGHT,"--dir_in",WAVELET_DIR,"--flistname_in",WAVELET_FILELIST,"--dir_out",LOG_WAVELET_SMA_DIR,"--flistname_out",LOG_WAVELET_SMA_FILELIST])
# 5-4. YULE-WALKER > SMA
#subprocess.run(["python3","4_sma.py","--data_len",YULE_LEN,"--wavelet_height",YULE_HEIGHT,"--dir_in",YULE_DIR,"--flistname_in",YULE_FILELIST,"--dir_out",YULE_SMA_DIR,"--flistname_out",YULE_SMA_FILELIST])
