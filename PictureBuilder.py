from tkinter import *
from tkinter import filedialog
from math import floor, ceil
from scipy import interpolate, ndimage
from remove_bd import *
from processing import do_processing_plot, set_energy_limits
import sys
import os
import pickle
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import PySimpleGUI as sg

REDCOLOR = '# '
GREENCOLOR = '№ '
PINKCOLOR = '@ '
NORMALCOLOR = ''

# Базовые параметры разметки спектра
freq = None
grate = None
rot180 = None
scale = None
graph_title = None
path = None

# Принимают значения при открытии нового файла (созданы для ввода в do_processing_all_files_in_a_folder из do_ask_open_file)
global_basename = None
global_filename = None

# Адреса папок
adress_of_home_dir = './'
address_of_last_dir_savefile = adress_of_home_dir + 'spectrograph_last_dir.pkl'
address_of_filters = adress_of_home_dir + 'Filters'
address_of_bd_map = adress_of_home_dir + 'bd_map.txt'
address_of_save_fig = adress_of_home_dir + 'Output pictures'
address_of_save_df = adress_of_home_dir + 'Output csv'
address_of_save_pkl = adress_of_home_dir + 'Output pkl'

# Параметры внешнего вида спектра
array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
data_frame = None
freq_step = 50
freq_from = 0
freq_to = 0
angle_from = 0
angle_to = 0
angle_step = 2
filters = {}
filters_number = 0
this_array_has_a_plot = False

normalize = True
patch_mode = False
angle_shift = 0
angle_rotate = 0
translate_rus = True
insert_title = True
show_pixels = False

if os.path.isfile(address_of_last_dir_savefile):
    with open(address_of_last_dir_savefile, 'rb') as dir_save_file:
        initdir = pickle.load(dir_save_file)
else:
    with open(address_of_last_dir_savefile, 'wb') as dir_save_file:
        pickle.dump('/', dir_save_file)


def update_progbar(window, i):
    window['progbar'].update_bar(i + 1)


def do_folder_preview(window, args):
    global data_frame, global_filename, global_basename, array, address_of_save_fig, address_of_save_df
    """Для всех файлов папки, где в последний раз был открыт файл, идет переконвертация (учитывая битые области, фильтры, поворот) в png.
    Работает с последним открытым типом файлов. """
    pathname = os.path.dirname(global_filename)
    filename_extension = os.path.splitext(global_filename)[-1]
    i = 1
    if (pathname):
        for file in os.listdir(pathname):
            event, values = window.read(timeout=0)
            if event == 'Cancel' or event is None:
                break
            update_progbar(window, int(i / len(os.listdir(pathname)) * 1000))
            i += 1
            if file.endswith(filename_extension):
                global_basename = file
                if file.endswith(".png"):
                    do_image_to_array('', pathname + "/" + file)
                elif file.endswith(".dat"):
                    do_data_to_array('', pathname + "/" + file)
                plt.close()
                preprocessing_plot()
                show_plot()
                fig = plt.gcf()
                fig.suptitle(global_basename[:global_basename.find("_")],
                             y=1, ha='right', fontsize=12)
                if (args):
                    data_frame.to_csv(address_of_save_df+'/' +
                                      global_basename.replace('.dat', '_csv.txt'), sep=' ')
                else:
                    plt.savefig(address_of_save_fig + '/' +
                                global_basename.replace('dat', 'png'), dpi=300)


def do_set_freq_limits(self, f):
    '''выбор пределов построения графика set(от [нм],до [нм]) ввод через пробел'''
    global freq_from, freq_to
    freq_from = int(f.split()[0])
    freq_to = int(f.split()[1])


def do_set_angle_limits(self, a):
    '''выбор пределов построения графика set(от [пикс],до [пикс]) ввод через пробел'''
    global angle_from, angle_to
    angle_from = int(a.split()[0])
    angle_to = int(a.split()[1])


def do_rotate_image(self, args=1):
    """Вращает на 90 градусов против часовой n раз. Количество поворотов обязательно"""
    global array
    array = np.rot90(array, k=int(args))


def do_hello(self, args):
    """Просто выводит 'hello world' на экран"""
    print("hello world")


def do_ask_add_filter(self, args):
    """Открытие GUI окна выбора файла для добавления"""
    global address_of_filters
    root = Tk()
    root.withdraw()
    root.option_add('*foreground', 'black')
    root.filename = filedialog.askopenfilename(
        initialdir=address_of_filters, title="Select file", filetypes=(("Filters", "*.txt"), ("All files", "*.*")))
    filename = root.filename
    if (type(filename) == str):
        filename_extension = os.path.splitext(filename)[-1]
        base = os.path.basename(filename)
        base_name = os.path.splitext(base)[0]
        if filename_extension == ".txt":
            do_list_push_filter(self='', name_of_file=base_name)
    root.destroy()


def do_list_push_filter(self, name_of_file):
    """Добавить новый фильтр. Принимает название файла без расширения. Ищет в папке Filters"""
    global filters, filters_number
    if (name_of_file != ''):
        filters_number += 1
        filters[filters_number] = name_of_file


def do_list_clear_filters(self, args):
    """Очистить список фильтров и сбросить их счетчик"""
    global filters, filters_number
    filters = {}
    filters_number = 0


def do_list_pop_filter(self, number):
    """Удалить последний фильтр. Принимает номер в списке <int>"""
    global filters, filters_number
    del filters[filters_number]
    filters_number -= 1


def do_save_filters(self, args):
    """Сохраняет список фильтров в файл spectrograph_filters.pkl в папку загруженного снимка"""
    global filters, path
    with open(path + 'spectrograph_filters.pkl', 'wb') as dir_save_file:
        pickle.dump(filters, dir_save_file)


def do_print_filters(self, args):
    """Выводит словарь фильтров"""
    global filters
    print("Filters: ", filters)


def do_ask_save_file(self, args):
    """Открытие GUI окна выбора файла для сохранения"""
    global plot, global_filename
    root = Tk()
    root.withdraw()
    root.option_add('*foreground', 'black')
    root.filename = filedialog.asksaveasfilename(initialdir="~", filetypes=(("PNG files only", "*.png"), ("All files", "*.*")),
                                                 initialfile=os.path.basename(os.path.dirname(global_filename)) + " " + os.path.split(os.path.splitext(global_filename)[-2])[-1]+".png")
    file_name = root.filename
    args.savefig(file_name, dpi=300)


def do_ask_open_file(self, reopen_without_asking_anything=False, this_filename=None):
    """Открытие GUI окна выбора файла для открытия"""
    global initdir, global_basename, global_filename
    root = Tk()
    root.withdraw()
    root.option_add('*foreground', 'black')
    if (reopen_without_asking_anything):
        if this_filename:
            root.filename = this_filename
        else:
            root.filename = global_filename
    else:
        root.filename = filedialog.askopenfilename(initialdir=initdir, title="Select file", filetypes=(
            ("Data files only", "*.dat"), ("PNG files only", "*.png"), ("All files", "*.*")))
    if (root.filename):
        global_filename = root.filename
        global_basename = os.path.basename(root.filename)
        filename_extension = os.path.splitext(root.filename)[-1]
        directory = os.path.dirname(root.filename)
        with open(address_of_last_dir_savefile, 'wb') as dir_save_file:
            pickle.dump(directory, dir_save_file)
        initdir = directory
        if filename_extension == ".png":
            do_image_to_array(self='', name_of_file=root.filename)
        elif filename_extension == ".dat":
            do_data_to_array(self='', name_of_file=root.filename)
        basepathname = os.path.basename(os.path.dirname(root.filename))
        open_new_file_with_gui = not reopen_without_asking_anything
        if this_filename or open_new_file_with_gui:
            do_set_parameters(self='', pathname=os.path.dirname(
                root.filename), dirname=basepathname)
    root.destroy()


def do_image_to_array(self, name_of_file):
    global array, this_array_has_a_plot
    this_array_has_a_plot = False
    array = plt.imread(name_of_file)


def do_data_to_array(self, name_of_file):
    global array, this_array_has_a_plot
    this_array_has_a_plot = False
    array = np.fromfile(name_of_file, dtype='>i2')
    array = np.reshape(array[4:], (array[1], array[3]))


class x_axis_frequency:
    def __init__(self):
        """Функция uз старых файлов Origin"""
        global array
        global grate, array, freq
        self.image_size = array.shape[1]
        if (grate == 300):
            self.offset = 1001
            self.dispersion = 0.12505
        elif (grate == 600):
            self.offset = 1011
            self.dispersion = 0.05918
        elif (grate == 900):
            self.offset = 1018
            self.dispersion = 0.03656
        else:
            print("Wrong grate")

        self.freq_array = [self.single(i) for i in range(0, self.image_size)]

    def single(self, i):
        global freq
        return (i - self.image_size + self.offset) * self.dispersion + freq

    def index(self, f):
        global freq
        # f=(i-image_size+offset)*dispersion+freq
        return round((f - freq) / self.dispersion + self.image_size - self.offset)

    def get_freq_unrounded(self):
        return self.freq_array

    def get_freq(self):
        return [round(i) for i in self.freq_array]


def get_angles():
    """Функция из старых файлов Origin"""
    global array, angle_shift
    image_size = array.shape[0]
    angle = [round(-0.0175 * (i - 1) + 11 +
                   angle_shift) for i in range(1, image_size + 1)]
    return angle


def get_angles_unrounded():
    """Функция из старых файлов Origin"""
    global array, angle_shift
    image_size = array.shape[0]
    angle = [-0.0175 * (i - 1) + 11 +
             angle_shift for i in range(1, image_size + 1)]
    return angle


def find_nearest(array, value):
    """Index of nearest"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def do_set_freq_step(self, args):
    """Выбор шага оси графика: set... <int>"""
    global freq_step
    freq_step = int(args)


def do_set_angle_step(self, args):
    """Выбор шага оси графика: set... <int>"""
    global angle_step
    angle_step = int(args)


def do_set_rotate(self, args):
    """Включение режима поворота на 180. ЗАПУСК БЕЗ АРГУМЕНТОВ ВЫСТАВЛЯЕТ ОТСУТСВИЕ ПОВОРОТА (также как и 0 False None)."""
    global rot180
    if (args and args != "0" and args != 'False' and args != 'None'):
        rot180 = True
    else:
        rot180 = False


def preprocessing_plot():
    global data_frame, angle_rotate, freq_from, freq_to, this_array_has_a_plot, plot, graph_title, rot180, freq_step, angle_step, array, scale, grate, filters, filters_number, normalize

    if (this_array_has_a_plot):
        do_ask_open_file(self='', reopen_without_asking_anything=True)
    this_array_has_a_plot = True

    # Подготовка массива к применению фильтров
    (bd_mult, bd_single) = read_bd_map(address_of_bd_map)
    apply_bd_map(array, bd_mult, bd_single)
    if (rot180):
        do_rotate_image(self='', args=2)

    width_of_background_borders = 10
    border_1 = np.mean(
        array[:width_of_background_borders, :width_of_background_borders],)
    border_2 = np.mean(
        array[-width_of_background_borders:, :width_of_background_borders])
    border_3 = np.mean(
        array[:width_of_background_borders, -width_of_background_borders:])
    border_4 = np.mean(
        array[-width_of_background_borders:, -width_of_background_borders:])
    sum_of_borders = [border_1, border_2, border_3, border_4]
    background = np.mean(sum_of_borders)
    background *= 1.02  # округление было вниз
    if (array[1, 1] > 1):  # -> dat -> type '>i2'
        background = background.astype('>i2')
    array -= background
    array[array < 0] = 0

    array = ndimage.rotate(array, angle_rotate, reshape=False)

    angle_array = get_angles()
    freq_class = x_axis_frequency()

    # Применение фильтров
    image_size = array.shape[1]
    array_factor = np.ones(image_size)
    do_list_push_filter("", name_of_file="Camera")
    if (grate == 300):
        do_list_push_filter("", name_of_file="300")
    elif (grate == 600):
        do_list_push_filter("", name_of_file="600")
    elif (grate == 900):
        do_list_push_filter("", name_of_file="900")

    for key, value in filters.items():
        filter_array = np.loadtxt(address_of_filters + '/' + value + '.txt')
        x = (filter_array[:, 0]).transpose()
        y = (filter_array[:, 1]).transpose()
        filter_function = interpolate.interp1d(x, y, fill_value="extrapolate")
        filter_vector_function = np.vectorize(filter_function)
        array_factor *= filter_vector_function(freq_class.get_freq())
    do_list_pop_filter('', len(filters))
    do_list_pop_filter('', len(filters))
    filters_number = filters_number - 2
    np.savetxt('filters.csv', array_factor)
    array_factor_reciprocal = np.reciprocal(array_factor)
    array_factor_rec_diag = np.diag(array_factor_reciprocal)
    array = array @ array_factor_rec_diag

    MAX = array.max()
    if (normalize):
        array *= 1.0 / MAX
    if (scale == 'log'):
        array[array <= 0] = np.exp(-10)
        array = np.log(array)


def show_plot():
    global show_pixels, data_frame, angle_from, angle_to, freq_from, freq_to, this_array_has_a_plot, plot, graph_title, rot180, freq_step, angle_step, array, scale, grate, filters, filters_number, patch_mode, translate_rus, insert_title
    fig, ax = plt.subplots()

    if (scale == 'log'):
        im = ax.imshow(array, cmap="nipy_spectral",
                       vmin=-5, vmax=0, aspect='auto')
    else:
        im = ax.imshow(array, cmap="nipy_spectral", aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    if (scale == 'log'):
        ctks = [-5, -4, -3, -2, -1, 0]
        ctkls = ["$e^{%d}$" % (v) for v in ctks[:]]
        cbar.set_ticks(ctks)
        cbar.set_ticklabels(ctkls)
    if translate_rus:
        cbar.ax.set_ylabel("Относительная интенсивность",
                           rotation=-90, va="bottom")
        ax.set_ylabel('Угол, мрад')
        ax.set_xlabel('Длина волны, нм')
    else:
        cbar.ax.set_ylabel("Relative intensity", rotation=-90, va="bottom")
        ax.set_ylabel('Angle, mrad')
        ax.set_xlabel('Wavelength, nm')

    if insert_title:
        ax.set_title(graph_title, fontsize=8)
    angle_array = get_angles()
    freq_class = x_axis_frequency()

    # Изменение меток на осях
    left, right = ax.get_xlim()
    if (freq_from and freq_to):  # обрезка изображения
        x_from = freq_class.index(freq_from)
        x_to = freq_class.index(freq_to)
        if not patch_mode:
            ax.set_xlim(x_from, x_to)
        left, right = x_from, x_to
    else:
        x_from = left
        x_to = right

    min_freq = freq_step * ceil(freq_class.single(left) / freq_step)
    max_freq = freq_step * floor(freq_class.single(right) / freq_step)
    new_label = range(min_freq, max_freq + freq_step, freq_step)
    new_tick = [freq_class.index(i) for i in new_label]
    ax.set_xticks(new_tick)
    ax.set_xticklabels(new_label)

    if not (angle_from or angle_to):
        angle_start = 0
        angle_finish = len(angle_array) - 1
    else:
        angle_start = angle_from
        angle_finish = angle_to
    min_angle = angle_step * ceil(angle_array[angle_finish] / angle_step)
    max_angle = angle_step * floor(angle_array[angle_start] / angle_step)
    new_label = range(min_angle, max_angle + angle_step, angle_step)
    new_tick = [find_nearest(angle_array, new_label[i])
                for i in range(0, len(new_label))]
    if not show_pixels:
        ax.set_yticks(new_tick)
        ax.set_yticklabels(new_label)
    if (angle_from or angle_to):
        if not patch_mode:
            ax.set_ylim(angle_to, angle_from)
    else:
        angle_from = 0
        angle_to = len(angle_array) - 1
    if (patch_mode):
        rect = patches.Rectangle((x_from, angle_from), (x_to - x_from),
                                 (angle_to - angle_from), linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_facecolor('black')
    fig.tight_layout()

    # complete pandas dataframe for future saving (see processing.py)
    # freq_class.single # ибо превышает
    # angle_array[i] # всегда меньше
    # array: angle_from [пиксель] angle_to [пикс] - от 0 до size()-1 по дефолту
    #      : x_from [пиксель] x_to [пиксель]

    if (x_from, x_to == ax.get_xlim()):  # это не баг, это фича не трожь
        x_from = int(x_from + 0.5)
        x_to = int(x_to - 0.5)
    data_frame_freq_array = [freq_class.single(i + x_from) for i in range(x_to - x_from + 1)]
    data_frame_angle_array = get_angles_unrounded()[angle_from:angle_to+1]
    data_frame_data_array = [[(array[j + angle_from][i + x_from] if ((i + x_from >= 0) and (i + x_from < 1920) and (j + angle_from >= 0)
                                                                     and (j + angle_from < 1200)) else 0) for i in range(x_to - x_from + 1)] for j in range(angle_to - angle_from + 1)]
    data_frame = pd.DataFrame(data_frame_data_array,
                              columns=data_frame_freq_array,
                              index=data_frame_angle_array)


def do_plot(self, args):
    """Открывает окно с графиком и текущими настройками в неблокирующем режиме"""
    global normalize, patch_mode, angle_shift, angle_rotate, translate_rus, insert_title, show_pixels

    normalize, patch_mode, angle_shift, angle_rotate, translate_rus, insert_title, show_pixels = args
    plt.close()
    preprocessing_plot()
    show_plot()


def do_exit(self, args):
    """Выход из работы. Альтернатива CTRL+c затем ENTER"""
    print(REDCOLOR + "\nЗавершение работы..." + NORMALCOLOR)
    sys.exit()


def do_print_array(self, args):
    """Выводит массив в текстовом режиме"""
    print(array)


def do_set_freq(self, args):
    """Изменят частоту на указанную set_freq <int>"""
    global freq
    freq = int(args)
    print(PINKCOLOR + "Frequency " + NORMALCOLOR + str(freq))


def do_set_grate(self, args):
    """Изменят решетку на указанную set_grate<int>"""
    global grate
    grate = int(args)
    print(PINKCOLOR + "Grating " + NORMALCOLOR + str(grate))


def do_set_scale(self, args):
    """Изменят шкалу на указанную set_scale <lin> or <log>"""
    global scale
    scale = args
    print(PINKCOLOR + "Scale " + NORMALCOLOR + scale)


def do_set_title(self, args):
    """Изменят заголовок на указанный set_title <str>"""
    global graph_title
    graph_title = args
    print(PINKCOLOR + "Title " + NORMALCOLOR + graph_title)


def do_print_parameters(self, args):
    """Вывод параметров в порядке аргументов у функции set"""
    global freq, grate, rot180, scale, graph_title, path
    param_turple = (freq, grate, rot180, scale, graph_title)
    print(param_turple)


def do_save_parameters(self, args):
    """Сохраняет параметры осей и заголовка в файл spectrograph_parameters.pkl в папку загруженного снимка"""
    global freq, grate, rot180, scale, graph_title, path
    param_turple = (freq, grate, rot180, scale, graph_title)
    with open(path + 'spectrograph_parameters.pkl', 'wb') as dir_save_file:
        pickle.dump(param_turple, dir_save_file)


def do_set_parameters(self, pathname="", frequency=0, grating=0, dirname=False, rotate=None, scaletype=False, title=None):
    global freq, grate, rot180, scale, graph_title, path, filters, filters_number
    print(PINKCOLOR + "Введенный/" + GREENCOLOR + 'сохраненный/' +
          REDCOLOR + "по умолчанию" + NORMALCOLOR + " параметр:")

    path = pathname
    if (pathname != ''):
        path = path + "/"
    if os.path.isfile(path + 'spectrograph_parameters.pkl'):
        with open(path + 'spectrograph_parameters.pkl', 'rb') as dir_save_file:
            param_turple = pickle.load(dir_save_file)
    else:
        param_turple = (None, None, None, None, None)
    if os.path.isfile(path + 'spectrograph_filters.pkl'):
        with open(path + 'spectrograph_filters.pkl', 'rb') as dir_save_file:
            filters = pickle.load(dir_save_file)
            filters_number = len(filters)
    # ---begin freq
    if (frequency):
        freq = frequency
        print(PINKCOLOR + "Frequency " + NORMALCOLOR + str(freq))
    elif (param_turple[0]):
        freq = param_turple[0]
        print(GREENCOLOR + "Frequency " + NORMALCOLOR + str(freq))
    else:
        freq = 800
        print(REDCOLOR + "Frequency " + NORMALCOLOR + str(freq))
    # ---begin grate
    if (grating):
        grate = grating
        print(PINKCOLOR + "Grating " + NORMALCOLOR + str(grate))
    elif (param_turple[1]):
        grate = param_turple[1]
        print(GREENCOLOR + "Grating " + NORMALCOLOR + str(grate))
    else:
        grate = 300
        print(REDCOLOR + "Grating " + NORMALCOLOR + str(grate))
    # ---begin rot (завязан на наличие частоты)
    if (rotate is not None):
        rot180 = rotate
        if (rotate):
            print(PINKCOLOR + "Rotate " + NORMALCOLOR + 'True')
        else:
            print(PINKCOLOR + "Rotate " + NORMALCOLOR + 'False')
    elif (param_turple[0]):
        rot180 = param_turple[2]
        if (param_turple[2]):
            print(GREENCOLOR + "Rotate " + NORMALCOLOR + 'True')
        else:
            print(GREENCOLOR + "Rotate " + NORMALCOLOR + 'False')
    else:
        rot180 = True
        print(REDCOLOR + "Rotate " + NORMALCOLOR + 'True')
    # ---begin scale
    if (scaletype):
        scale = scaletype
        print(PINKCOLOR + "Scale " + NORMALCOLOR + scale)
    elif (param_turple[3]):
        scale = param_turple[3]
        print(GREENCOLOR + "Scale " + NORMALCOLOR + scale)
    else:
        scale = 'lin'
        print(REDCOLOR + "Scale " + NORMALCOLOR + scale)
    # ---begin title
    if (title):
        graph_title = title
        print(PINKCOLOR + "Title " + NORMALCOLOR + graph_title)
    elif (dirname and dirname != '..'):
        graph_title = dirname
        print(PINKCOLOR + "Title (dir) " + NORMALCOLOR + graph_title)
    elif (param_turple[4]):
        graph_title = param_turple[4]
        print(GREENCOLOR + "Title " + NORMALCOLOR + graph_title)
    else:
        graph_title = "Частотно-угловой спектр филамента"
        print(REDCOLOR + "Title " + NORMALCOLOR + grath_title)
    # ---begin filters
    do_print_filters(self='', args='')
