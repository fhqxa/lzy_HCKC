import os
import shutil
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # linux
import matplotlib.pyplot as plt

import utility0


flag_x = 0

dict_plot_style = {
    '0_0': 'black',       '0_1': '-',
    '1_0': 'darkgray',    '1_1': '--',
    '2_0': 'orange',      '2_1': '-.',
    '3_0': 'lime',        '3_1': ':',
    '4_0': 'deepskyblue', '4_1': '-',
    '5_0': 'blue',        '5_1': '-',
}
style = 0
len_style = len(dict_plot_style) / 2

log_parent = os.getcwd()  # get the parent directory of the current 'py' file
save_parent = log_parent
for log0 in os.listdir(log_parent):  # in 'project1/'
    if 'log' in log0:
        save0 = os.path.join(save_parent, 'analyze')
        shutil.rmtree(save0) if os.path.exists(save0) else ''

        log0 = os.path.join(log_parent, log0)
        for log_Dataset in os.listdir(log0):  # in 'log/'
            if 'log' in log_Dataset:
                save_Dataset = os.path.join(save0, log_Dataset)
                log_Dataset = os.path.join(log0, log_Dataset)
                for log in os.listdir(log_Dataset):  # in 'log_Cifar100_vision_LongTail/'
                    if 'log' in log:
                        save = os.path.join(save_Dataset, log)
                        log = os.path.join(log_Dataset, log)
                        for log_train in os.listdir(log):  # in 'log_1/'
                            if 'log_' in log_train:
                                save_train = os.path.join(save, log_train)
                                os.makedirs(save_train) if not os.path.exists(save_train) else ''

                                log_train = os.path.join(log, log_train)
                                for sw_itself in os.listdir(log_train):  # in 'log_coarse/'
                                    if 'sw' in sw_itself:
                                        sw = os.path.join(log_train, sw_itself)
                                        for file_csv_itself in os.listdir(sw):  # in 'sw/'
                                            if 'csv' in file_csv_itself:
                                                file_csv = os.path.join(sw, file_csv_itself)

                                                id_name = file_csv_itself.split('log_')[1].split('.csv')[0]
# ==========================================================================================

                                                L_c_ = utility0.ColumnFromCsv('L_c_', file_csv)
                                                L_c = utility0.ColumnFromCsv('L_c', file_csv)

                                                L_c = np.array([float(i) for i in L_c])
                                                L_c_ = np.array([float(i) for i in L_c_])

                                                if flag_x == 0:
                                                    plt.figure(figsize=(6, 6), dpi=300)
                                                    x = np.linspace(0, len(L_c) - 1, num=len(L_c), dtype=int)
                                                    flag_x = 1

                                                for d in list(dict_plot_style.items()):
                                                    key, value = d[0], d[1]
                                                    if f'{style}_' in key:
                                                        if f'_0' in key:
                                                            color = value
                                                        if f'_1' in key:
                                                            linestyle = value
                                                style += 1 if style < len_style else -len_style

                                                plt.plot(x, L_c_, label=id_name, color=color, linestyle=linestyle)
                                                plt.plot(x, L_c, color=color, linestyle=linestyle)

plt.yticks(np.linspace(0.0, 5.0, num=26))
plt.grid()
# plt.ylabel('')
plt.xlabel('Epoch')
# plt.title('')
plt.legend(loc='lower left')  # 注释栏
plt.savefig(save_train + '/plot_L_c')  # 覆盖同名文件
# plt.show()
plt.close()







