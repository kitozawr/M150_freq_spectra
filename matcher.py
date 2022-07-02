import os
import pickle
import struct

import bottleneck as bn
import matplotlib.pylab as plt
import numpy as np
from scipy.signal import find_peaks


# %% Чтение бинарных файлов с нового осциллографа.
def read_bin_new_Rudnev(filepath):
    with open(filepath, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read()
    # E = struct.unpack('>l', data[0:4])[0]
    T = (struct.unpack('15s', data[4:19])[0]).decode('UTF-8')
    dt = struct.unpack('>d', data[19:27])[0]
    ch0_on = struct.unpack('?', data[27:28])[0]  # whether channel 0 was on
    dV0 = struct.unpack('>d', data[28:36])[0]  # [mV/bit]
    # ch0_adj = struct.unpack('>d', data[36:44])[0]  # channel0 adjustment
    ch1_on = struct.unpack('?', data[44:45])[0]  # whether channel 1 was on
    dV1 = struct.unpack('>d', data[45:53])[0]  # [mV/bit]
    # ch1_adj = struct.unpack('>d', data[53:61])[0]  # channel0 adjustment

    wf_len = len(data) - 61
    s = '>b' + 'b' * (wf_len - 1)

    waveform = np.fromiter(struct.unpack(s, data[61:]), dtype='int8')

    if ch0_on and ch1_on:
        # ch_num = 2
        ch0_arr = waveform[0::2] * dV0
        ch1_arr = waveform[1::2] * dV1
        ch0_arr[1] = 0
        return T, dt, ch0_arr, ch1_arr
    elif ch0_on:
        # ch_num = 0
        ch0_arr = waveform * dV0
        ch1_arr = ch0_arr
        ch0_arr[1] = 0
        return T, dt, ch0_arr, ch1_arr
    elif ch1_on:
        # ch_num = 1
        ch1_arr = waveform * dV1
        ch0_arr = ch1_arr
        ch0_arr[1] = 0
        return T, dt, ch0_arr, ch1_arr
    else:
        print("ERROR. Unknown channels configuration in read_bin_new_Rudnev in data_proc_basics_script")
        return "ERROR"


def calc_lum_time(string):
    """
    Calculate time from a time string in format h_m_s,ms.
    """

    if ',' in string:
        string_splitted, ms = string.split(",")
    elif '.' in string:
        string_splitted, ms = string.split(".")
    elif string.isdecimal():
        return int(string)
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
    return time


def make_file_list_to_compare_new_program(pathname_ac, ext):
    """
    Returns array of times in ms, corresponding to files and the files to be compared with energies.
    Arrays are sorted ascending by time.
    Parameters:
        pathname_ac - folder with the files to be compared,
        ext. Allowed values: '.dat', '.bin', '.tif'. In any other case the function returns 1.
    """

    if (ext != '.dat') and (ext != '.png') and (ext != '.RAW') and (ext != '.bin') and (ext != '.tif'):
        print('In module "data_proc_basics", function "make_file_list_to_compare_new_program":')
        print("ERROR: unknown data type!")
        return [], []
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
    if not filenames_ac:
        print("Acoustcs folder is empty.")
        return [], []
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


def subtract_plane(data):
    """
    Function subtracts an inclined plane from data background.
    """

    # Constants.
    stripe_width = 5  # Ширина полосы по краям кадра (в пикселях), используемая для расчёта параметров вычитаемой
    # плоскости.

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

    return data


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
    return data, width, height


def run_av(array, av_size):
    average = np.zeros(len(array) - av_size)
    for i in range(0, len(array) - av_size):
        average[i] = np.average(array[i:i + av_size])
    return average


def find_peak_acoustic(wf, dt=None):
    # %% Константы
    t_start_num = 5  # Отсуп на графике в начале waveform (в связи с наличием провала в начале).
    dt_between_max = 10e-6  # Нижняя граница расстояния между двумя максимумами на wf.
    SHIFT_LEFT = 37.75e-6  # Сдвиг левой границы области усреднения (1-го максимума) относительно главного максимума.
    SHIFT_RIGHT = 30.75e-6  # Сдвиг правой границы области усреднения (1-го максимума) относительно главного максимума.
    DIST_BETW_MAX = 26.74e-6  # Сдвиг между двумя самыми высокими максимумами.
    DIST_BETW_MAX_EPS = 0.05  # Ширина коридора между максимумами в долях расстояния.
    MAX_DIFF_BETWEEN_THE_MAXIMA = 3  # Найденные "большие" максимумы могут отличаться не более чем в это количество раз,
    # иначе один из них - паразитный.
    dt_av = 1.0e-6  # Диапазон усреднения для бегущего среднего.
    dt_av_for_max_pick = 2e-6  # Диапазон усреднения для бегущего среднего для поиска главных максимумов.
    FON_SHIFT = 10e-6  # Сдвиг левой границы относительно shift left для подсчёта фона.
    V_to_J = 100.0  # Коэффициент пересчёта из вольт в Джоули.
    threshold = 0.45  # Энергии ниже этого уровня не учитываются при калибровке. Default is 0.45.

    # %% Поиск максимумов
    N_elements_in_the_moving_window = int(round(dt_av_for_max_pick / dt))
    maxima, _ = find_peaks(bn.move_mean(wf[t_start_num:],
                                        N_elements_in_the_moving_window)[N_elements_in_the_moving_window - 1:],
                           distance=int(round(dt_between_max / dt)))
    maxima = sorted(maxima, key=lambda x: wf[x])[-2:]
    # print(maxima)
    # print(int(round(SHIFT_LEFT / dt)))
    dist_betw_max = abs(maxima[1] - maxima[0]) * dt
    if (dist_betw_max < DIST_BETW_MAX * (1 + DIST_BETW_MAX_EPS)) and (
            dist_betw_max > DIST_BETW_MAX * (1 - DIST_BETW_MAX_EPS)) and (
            wf[maxima[0]] * MAX_DIFF_BETWEEN_THE_MAXIMA > wf[maxima[1]]):
        maxima = np.sort(maxima)
        max_coord = maxima[1]  # Координата последнего из двух самых высоких максимумов.
        shift_left = SHIFT_LEFT
        shift_right = SHIFT_RIGHT
    else:
        max_coord = maxima[1]  # Координата самого высокого максимума, считаем, что второй не поместился.
        shift_left = SHIFT_LEFT - DIST_BETW_MAX
        shift_right = SHIFT_RIGHT - DIST_BETW_MAX

    if max_coord <= int(round(shift_left / dt)):  # Проверка того, что максимум не слишком близко к началу выборки.
        return None
    # %% Пересчёт сдвигов из секунд в отсчёты.
    shift_left_num = int(round(shift_left / dt))
    shift_right_num = int(round(shift_right / dt))
    fon_shift_num = int(round(FON_SHIFT / dt))

    # Установка левой границы, правой границы, и границы области для вычисления нулевого значения.
    left_border = max_coord - shift_left_num
    right_border = max_coord - shift_right_num
    if left_border - fon_shift_num >= 0:
        fon_left_border = left_border - fon_shift_num
    else:
        fon_left_border = 0

    dnum_av = int(round(dt_av / dt))  # Ширина полосы для бегущего среднего в отсчётах.
    average = bn.move_mean(wf[left_border:right_border + 1], dnum_av)[dnum_av - 1:]  # Вычисление бегущего среднего.
    en_V = np.amax(average)  # Энергия в вольтах без учёта фона.
    fon = np.mean(wf[fon_left_border:left_border])
    energy = (en_V - fon) * V_to_J
    if energy < threshold:
        return None
    return energy


def find_kalibr(pathname_kalibr, path_to_save=None):
    time_en = np.zeros(len(os.listdir(pathname_kalibr)))
    energies_0 = np.zeros(len(os.listdir(pathname_kalibr)))
    energies_1 = np.zeros(len(os.listdir(pathname_kalibr)))
    for i, filename_en_inner in enumerate(os.listdir(pathname_kalibr)):
        if filename_en_inner.endswith(".bin"):
            T, dt, wf0, wf1 = read_bin_new_Rudnev(pathname_kalibr + "/" + filename_en_inner)
            energies_0[i] = np.amax(wf0)
            energies_1[i] = find_peak_acoustic(wf1, dt=dt)
            time_en_list = T.replace(',', '.').split("-")
            time_en[i] = float(time_en_list[-1]) + 60 * float(time_en_list[-2]) + 3600.0 * float(time_en_list[-3])
            time_en[i] = time_en[i] if energies_1[i] > 0 else -1
    # time_en[i]<0 - универсальный маркер ошибочных данных
    A = np.vstack([energies_0[time_en > 0], np.ones(len(energies_0[time_en > 0]))]).T
    try:
        m, c = np.linalg.lstsq(A, energies_1[time_en > 0], rcond=None)[0]
    except:
        m, c = -1, 0
    plt.figure()
    fig, ax = plt.subplots()
    plt.scatter(energies_0[time_en > 0], energies_1[time_en > 0])
    ax.legend(['m=' + str(m) + " c=" + str(c)])
    ax.axline((1, 1 * m + c), slope=m, color='C0')
    plt.title(pathname_kalibr, fontsize=8)
    import PictureBuilder as PB
    if path_to_save:
        filename = path_to_save + '/kalibr' + '.png'
    else:
        filename = PB.address_of_save_fig + '/kalibr' + '.png'
    plt.savefig(filename, dpi=100)
    with open(pathname_kalibr + '/kalibr.pkl', 'wb') as kalibr_save_file:
        pickle.dump((m, c), kalibr_save_file)
    return m, c


def make_dictionary(pathname):
    pathname_en = pathname.replace('Спектры', 'Энергии')
    if os.path.exists(pathname_en + '/TestFolder'):
        pathname_en = pathname_en + '/TestFolder'
    pathname_ac = pathname.replace('Спектры', 'Моды')
    pathname_pb = pathname
    if os.path.exists(pathname_en):
        time_en = np.zeros(len(os.listdir(pathname_en)))
        for i, filename_en_inner in enumerate(os.listdir(pathname_en)):
            if filename_en_inner.endswith(".bin"):
                T, dt, wf0, wf1 = read_bin_new_Rudnev(pathname_en + "/" + filename_en_inner)
                time_en_list = T.replace(',', '.').split("-")
                time_en[i] = float(time_en_list[-1]) + 60 * float(time_en_list[-2]) + 3600.0 * float(time_en_list[-3])
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
    for i, filename_pb_inner in enumerate(os.listdir(pathname_pb)):
        if filename_pb_inner.endswith(".dat"):
            time_pb[i] = (int(filename_pb_inner.split("_")[-3]) * 3600 + int(
                filename_pb_inner.split("_")[-2]) * 60 + float(filename_pb_inner.split("_")[-1][0:6].replace(",", ".")))
    dictionary = {}
    for i in range(0, len(time_pb)):
        index_en = np.argmin(np.abs(time_pb[i] - time_en))
        index_mode = np.argmin(np.abs(time_pb[i] - time_mode))
        if time_mode[index_mode] < time_pb[i]:
            index_mode = index_mode + 1
        if index_mode == len(time_mode):
            index_mode = -1

        cond1 = np.abs(time_pb[i] - time_mode[index_mode]) < 0.05
        cond2 = np.abs(time_pb[i] - time_en[index_en]) < 0.05
        if not cond1:
            index_mode = -1
        if not cond2:
            index_en = -1
        dictionary[i] = (index_mode, index_en)
        with open(pathname + '/dictionary_of_match.pkl', 'wb') as dict_save_file:
            pickle.dump(dictionary, dict_save_file)
    return dictionary
