import os
import struct

import matplotlib.pylab as plt
import numpy as np
import pywt


def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j


def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


# %% Чтение бинарных файлов с нового осциллографа.
def read_bin_new_Rudnev(filepath):
    with open(filepath, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read()
    E = struct.unpack('>l', data[0:4])[0]
    T = (struct.unpack('15s', data[4:19])[0]).decode('UTF-8')
    dt = struct.unpack('>d', data[19:27])[0]
    ch0_on = struct.unpack('?', data[27:28])[0]  # whether channel 0 was on
    dV0 = struct.unpack('>d', data[28:36])[0]  # [mV/bit]
    ch0_adj = struct.unpack('>d', data[36:44])[0]  # channel0 adjustment
    ch1_on = struct.unpack('?', data[44:45])[0]  # whether channel 1 was on
    dV1 = struct.unpack('>d', data[45:53])[0]  # [mV/bit]
    ch1_adj = struct.unpack('>d', data[53:61])[0]  # channel0 adjustment

    wf_len = len(data) - 61
    s = '>b' + 'b' * (wf_len - 1)

    waveform = np.fromiter(struct.unpack(s, data[61:]), dtype='int8')

    # ch0_arr = [dV, ch0_adj, wf0]
    # ch1_arr = [dV, ch1_adj, wf1]

    # print("filename = {3}, T = {0}, dt = {1}, dV = {2}".format(T, dt, dV0, dV1, filepath))

    if ch0_on and ch1_on:
        ch_num = 2
        ch0_arr = waveform[0::2] * dV0
        ch1_arr = waveform[1::2] * dV1
        ch0_arr[1] = 0
        return (T, dt, ch0_arr, ch1_arr)
    elif ch0_on:
        ch_num = 0
        ch0_arr = waveform * dV0
        ch1_arr = ch0_arr
        ch0_arr[1] = 0
        return (T, dt, ch0_arr, ch1_arr)
    elif ch1_on:
        ch_num = 1
        ch1_arr = waveform * dV1
        ch0_arr = ch1_arr
        ch0_arr[1] = 0
        return (T, dt, ch0_arr, ch1_arr)
    else:
        print("ERROR. Unknown channels configuration in read_bin_new_Rudnev in data_proc_basics_script")
        return ("ERROR")



def calc_lum_time(string):
    '''
    Calculate time from a time string in format h_m_s,ms.
    '''

    if ',' in string:
        string_splitted, ms = string.split(",")
    elif '.' in string:
        string_splitted, ms = string.split(".")
    elif string.isdecimal():
        return (int(string))
    else:
        print("ERROR: invalid string in calc_lum_time.\nString: \"{}\"".format(string))
        raise TypeError("Invalid string in calc_lum_time.\nString: \"{}\"".format(string))
    digit_num = len(ms)
    if digit_num >= 3:
        ms = round(int(ms) / 10 ** (digit_num - 3))
    if '_' in string_splitted:
        h, m, s = [float(f) for f in string_splitted.split("_")]
    else:
        h, m, s = [float(f) for f in string_splitted.split("-")]
    time = h * 3600.0 + m * 60.0 + s + ms * 0.001
    return (time)


def make_file_list_to_compare_new_program(pathname_ac, ext):
    '''
    Returns array of times in ms, corresponding to files and the files to be compared with energies.
    Arrays are sorted ascending by time.
    Parameters:
        pathname_ac - folder with the files to be compared,
        ext. Allowed values: '.dat', '.bin', '.tif'. In any other case the function returns 1.
    '''

    if (ext != '.dat') and (ext != '.png') and (ext != '.RAW') and (ext != '.bin') and (ext != '.tif'):
        print('In module "data_proc_basics", function "make_file_list_to_compare_new_program":')
        print("ERROR: unknown data type!")
        return ([], [])
    # Формирование несортированного списка файлов с акустикой.
    if ext == '.tif':
        filenames_ac = [f for f in os.listdir(pathname_ac) if (f.endswith(ext) and "fil" in f)]
    elif ext == '.png':
        filenames_ac = [f for f in os.listdir(pathname_ac) if f.endswith(ext) and '__' in f]
    elif ext == '.RAW':
        filenames_ac = [f for f in os.listdir(pathname_ac) if f.endswith(ext) and 'MV-UB130GM' in f]
    else:
        filenames_ac = [f for f in os.listdir(pathname_ac) if f.endswith(ext)]
    # Если папка пустая, сразу возвращаем пустые списки (поиск скачка даёт ошибку при пустых списках).
    if filenames_ac == []:
        print("Acoustcs folder is empty.")
        return ([], [])
    if ext == '.RAW':
        filenames_ac_info = [f.split(ext)[0].split("-") for f in filenames_ac]
    else:
        filenames_ac_info = [f.split(ext)[0].split("__") for f in filenames_ac]
    filenames_ac_info_ext = []

    for f in filenames_ac_info:
        if ext == '.RAW':
            segment = f[-1]
        else:
            for segment in f:
                if ('.' in segment or ',' in segment) and '-' in segment:
                    break
        time = calc_lum_time(segment)
        filenames_ac_info_ext.append([time, f])
    filenames_ac_info_ext = sorted(filenames_ac_info_ext, key=lambda x: float(
        x[0]))  # Cортированный по первому элементу названия (времени в мс) список файлов с акустикой.
    filenames_ac_times = np.array([int(round(f[0] * 1000)) for f in filenames_ac_info_ext])
    filenames_ac_info = [f[1] for f in filenames_ac_info_ext]

    return filenames_ac_times, filenames_ac_info


def subtract_plane(data, quite=False):
    '''
    Function subtracts an inclined plane from data background.
    '''

    # Constants.
    stripe_width = 5  # Ширина полосы по краям кадра (в пикселях), используемая для расчёта параметров вычитаемой плоскости.

    # Выделяем полосы вдоль сторон массива шириной stripe_width без повторения элементов.
    data1 = data[:stripe_width, ].flatten()
    data2 = data[stripe_width:-stripe_width, :stripe_width].flatten()
    data3 = data[stripe_width:-stripe_width, -stripe_width:].flatten()
    data4 = data[-stripe_width:, ].flatten()

    values = np.hstack((data1, data2, data3, data4))  # Cоединяем значения выбранных элементов в один одномерный массив.

    # Готовим массив X ординат (номеров) элементов из массива values.
    x1 = np.tile(np.arange(0, data.shape[1]), stripe_width)
    x2 = np.tile(np.arange(0, stripe_width), data.shape[0] - 2 * stripe_width)
    x3 = np.tile(np.arange(data.shape[1] - stripe_width, data.shape[1]), data.shape[0] - 2 * stripe_width)
    x4 = np.tile(np.arange(0, data.shape[1]), stripe_width)

    X = np.hstack((x1, x2, x3, x4))

    # Готовим массив Y ординат (номеров) элементов из массива values.
    y1 = np.repeat(np.arange(0, stripe_width), data.shape[1])
    y2 = np.repeat(np.arange(stripe_width, data.shape[0] - stripe_width), stripe_width)
    y3 = np.repeat(np.arange(stripe_width, data.shape[0] - stripe_width), stripe_width)
    y4 = np.tile(np.arange(data.shape[0] - stripe_width, data.shape[0]), data.shape[1])

    Y = np.hstack((y1, y2, y3, y4))

    Z = np.ones_like(X)
    coords = np.vstack((X, Y, Z)).T  # Массив с "правильной" (с т.зр. перемножения матриц) размерностью.

    A, B, C = np.linalg.lstsq(coords, values, rcond=None)[0]  # МНК

    I = np.arange(0, data.shape[1])
    J = np.arange(0, data.shape[0])
    ii, jj = np.meshgrid(I, J, indexing='xy')

    plane_data = A * ii + B * jj + C
    data = data - plane_data

    return (data)


def read_raw_Mind_Vision(filename):
    """
    Function opens .RAW file from chineese CCD. The data should be in format of 16-bit raw matrix.
    Parameters:
    -------------------------------------------
    - filename : path (string)
        Path to the file to read data from.
    """

    # Constants.
    width = 1280  # Длина кадра.
    height = 960  # Ширина кадра.
    writing_gain = 16  # Коэффициент, на которые нужно разделить данные для получения исходных значений.

    # binary data file reading
    with open(filename, "rb") as binary_file:
        data_bin = binary_file.read()

    f_size = width * height

    try:
        s = '<H' + 'H' * (f_size - 1)
        data = np.fromiter(struct.unpack(s, data_bin), dtype='uint16')
    except struct.error:
        print("In data_proc/lumin_proc, in function 'read_raw_Mind_Vision':")
        print("ERROR: could not read data file {}".format(filename))
        return None

    data = np.reshape(data, (height, width))
    data = data / writing_gain  # Devide by 'gain' introduced by writing of the 12-bit image into 16 bit.

    return (data, width, height)


def find_kalibr(pathname_kalibr, path_to_save= None):
    time_en = np.zeros(len(os.listdir(pathname_kalibr)))
    energies_0 = np.zeros(len(os.listdir(pathname_kalibr)))
    energies_1 = np.zeros(len(os.listdir(pathname_kalibr)))
    for i, filename_en_inner in enumerate(os.listdir(pathname_kalibr)):
        if filename_en_inner.endswith(".bin"):
            T, dt, wf0, wf1 = read_bin_new_Rudnev(pathname_kalibr + "/" + filename_en_inner)
            time_en_list = T.replace(',', '.').split("-")
            time_en[i] = float(time_en_list[-1]) + 60 * float(time_en_list[-2]) + 3600.0 * float(time_en_list[-3])
            energies_0[i] = np.amax(wf0)
            energies_1[i] = np.amax(wf1)
    A = np.vstack([energies_0[time_en > 0], np.ones(len(energies_0[time_en > 0]))]).T
    m, c = np.linalg.lstsq(A, energies_1[time_en > 0] * 100, rcond=None)[0]
    #print(m, c)
    fig = plt.figure()
    plt.scatter(energies_0[time_en > 0], energies_1[time_en > 0] * 100)
    plt.title(pathname_kalibr, fontsize=8)
    import PictureBuilder as PB
    if path_to_save:
        filename = path_to_save + '/kalibr' + '.png'
    else:
        filename = PB.address_of_save_fig + '/kalibr'+ '.png'
    plt.savefig(filename, dpi=100)
    # plt.plot(time_en[time_en > 0], )
    # plt.show()
    return (m, c)


def make_dictionary(pathname):
    pathname_en = pathname.replace('Спектры', 'Энергии')
    if (os.path.exists(pathname_en+'/TestFolder')):
        pathname_en = pathname_en+'/TestFolder'
    # print(pathname_en)
    pathname_ac = pathname.replace('Спектры', 'Моды')
    pathname_pb = pathname
    if os.path.exists(pathname_en):
        time_en = np.zeros(len(os.listdir(pathname_en)))
        # level_en = np.zeros(len(os.listdir(pathname_en)))
        for i, filename_en_inner in enumerate(os.listdir(pathname_en)):
            if filename_en_inner.endswith(".bin"):
                T, dt, wf0, wf1 = read_bin_new_Rudnev(pathname_en + "/" + filename_en_inner)
                time_en_list = T.replace(',', '.').split("-")
                time_en[i] = float(time_en_list[-1]) + 60 * float(time_en_list[-2]) + 3600.0 * float(time_en_list[-3])
                # level_en[i] = np.amax(wf0)
    else:
        time_en = np.zeros(1)

    if os.path.exists(pathname_ac):
        ext = '.RAW'
        filenames_ac_times, filenames_ac_info = make_file_list_to_compare_new_program(pathname_ac, ext)
        time_mode = np.zeros(len(filenames_ac_times))

        nonzero_shift = True
        shift = 0
        step = 1
        while nonzero_shift:
            nonzero_shift = False
            # print(shift)
            for i in range(0, len(filenames_ac_times)):
                time_mode[i] = shift + (
                        float(filenames_ac_info[i][-1]) - float(filenames_ac_info[0][-1])) / 1e7 + float(
                    filenames_ac_info[0][-2][-2:]) + 60 * float(filenames_ac_info[0][-2][-4:-2]) + 3600 * float(
                    filenames_ac_info[0][-2][-6:-4])
                if (int(time_mode[i]) < int(filenames_ac_info[i][-2][-2:]) + 60 * int(
                        filenames_ac_info[i][-2][-4:-2]) + 3600 * int(filenames_ac_info[i][-2][-6:-4])):
                    shift = shift + 1 / (2 ** step)
                    step = step + 1
                    nonzero_shift = True
                if (int(time_mode[i]) > int(filenames_ac_info[i][-2][-2:]) + 60 * int(
                        filenames_ac_info[i][-2][-4:-2]) + 3600 * int(filenames_ac_info[i][-2][-6:-4])):
                    shift = shift - 1 / (2 ** step)
                    step = step + 1
                    nonzero_shift = True
    else:
        time_mode = np.zeros(1)

    time_pb = np.zeros(len(os.listdir(pathname_pb)))
    # level_pb = np.zeros(len(os.listdir(pathname_pb)))
    for i, filename_pb_inner in enumerate(os.listdir(pathname_pb)):
        if filename_pb_inner.endswith(".dat"):
            time_pb[i] = (int(filename_pb_inner.split("_")[-3]) * 3600 + int(
                filename_pb_inner.split("_")[-2]) * 60 + float(filename_pb_inner.split("_")[-1][0:6].replace(",", ".")))
            # from PictureBuilder import do_data_to_array, preprocessing_plot
            # do_data_to_array('', pathname + "/" + filename_pb_inner)
            # preprocessing_plot(raw_output=True)
            # from PictureBuilder import array, angle_from, angle_to
            # level_pb[i] = array[angle_from:angle_to + 1, 0: 1900].mean()
            # print(time_pb[i])

    dictionary = {}
    # plt.scatter(time_en, np.ones(len(time_en)))
    # plt.scatter(time_mode, 2*np.ones(len(time_mode)))
    # plt.scatter(time_pb, np.zeros(len(time_pb)))
    # plt.show()
    # print(time_en)
    for i in range(0, len(time_pb)):
        index_en = np.argmin(np.abs(time_pb[i] - time_en))
        index_mode = np.argmin(np.abs(time_pb[i] - time_mode))
        if time_mode[index_mode] < time_pb[i]:
            index_mode = index_mode + 1
        # print(index_mode, len(time_mode))
        if index_mode == len(time_mode):
            index_mode = -1

        cond1 = np.abs(time_pb[i] - time_mode[index_mode]) < 0.05
        cond2 = np.abs(time_pb[i] - time_en[index_en]) < 0.05
        if not cond1:
            index_mode = -1
        if not cond2:
            index_en = -1
        dictionary[i] = (index_mode, index_en)
        # print(i, index_mode, index_en)
        # print(str(i) + " " + str(np.argmin(np.abs(time_en[i] - time_mode)))+" "+str(np.argmin(np.abs(time_en[i] - time_pb))))
    # best_shift_en = 0
    # best_match_value = 0
    # for i in range(0, len(time_pb)):
    #     index_en = np.argmin(np.abs(time_pb[i] - time_en))
    #     index_mode = np.argmin(np.abs(time_pb[i] - time_mode))
    #     if time_mode[index_mode] < time_pb[i]:
    #         index_mode = index_mode + 1
    #     print(i, dictionary[i][0], dictionary[i][1], time_pb[i], time_mode[index_mode], time_en[index_en])
    # for shift_en in range(-4, 5):
    #     level_en_for_pb = [level_en[dictionary.get(i)[1]+shift_en] for i in range(len(level_pb))]
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(level_pb[:-1], level_en_for_pb[:-1])
    #     if r_value**2 > best_match_value:
    #         best_shift_en = shift_en
    #         best_match_value = r_value**2
    #     print(shift_en, slope, intercept, r_value**2, p_value, std_err)
    # for i in range(len(level_pb)-1):
    #     try:
    #         if 0.5 < best_match_value:
    #             dictionary[i] = (dictionary.get(i)[1]+best_shift_en, dictionary.get(i)[2])
    #         else:
    #             dictionary[i] = (-1, dictionary.get(i)[2])
    #     except:
    #         pass
    # print(best_shift_en)
    return dictionary

