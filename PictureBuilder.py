from tkinter import *
from tkinter import filedialog
from math import floor, ceil
from scipy import interpolate
import sys
import os
import pickle
import matplotlib.pylab as plt
import seaborn as sns; sns.set()
import numpy as np
from PictureBuilder.remove_bd import *


REDCOLOR = '\033[1;31;40m'
GREENCOLOR = '\033[0;32;47m'
PINKCOLOR= '\033[1;35;40m'
NORMALCOLOR =  '\033[0m'

freq=None
grate=None
rot180=None
scale=None
graph_title=None
path=None
basename=None
filename=None

adress_of_home_dir='/home/student/Desktop/PictureBuilder/'

address_of_last_dir_savefile= adress_of_home_dir+'spectrograph_last_dir.pkl'
address_of_filters= adress_of_home_dir+'Filters'
address_of_bd_map= adress_of_home_dir+'bd_map.txt'
address_of_save_fig= adress_of_home_dir+ 'Saves'

array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
freq_step=50
freq_from= 0
freq_to=   0
angle_step=2
filters={}
filters_number=0
this_array_has_a_plot= False

if os.path.isfile(address_of_last_dir_savefile):
    with open(address_of_last_dir_savefile,'rb') as dir_save_file:
        initdir= pickle.load(dir_save_file)
else:
    with open(address_of_last_dir_savefile,'wb') as dir_save_file:
        pickle.dump('/', dir_save_file)

def do_set_freq_limits (self,f):
    #выбор пределов построения графика set(от [нм],до [нм])
    freq_from=f[0]
    freq_to=  f[1]
       
def do_processing_all_files_in_a_folder(self,args):
    """Для всех файлов папки, где в последний раз был открыт файл, идет переконвертация (битые области, фильтры, поворот) сырых данных в готовый массив для дальнейшей обработки"""
    global array, filename
    pathname=os.path.dirname(filename)
    if (pathname):
        for file in os.listdir("pathname"):
            if file.endswith(".png"):
                do_image_to_array('', file)
            elif file.endswith(".dat"):
                do_data_to_array('', file)
            do_plot(self='', args='no_plot')
        
def do_rotate(self, args=1):
    """Вращает на 90 градусов против часовой n раз. Количество поворотов обязательно"""
    global array
    array=np.rot90(array, k=int(args))

def do_hello(self, args):
    """Просто выводит 'hello world' на экран"""
    print ("hello world")

def do_ask_add_filter(self,args):
    """Открытие GUI окна выбора файла для добавления"""
    global address_of_filters
    root = Tk()
    root.withdraw()
    root.filename =  filedialog.askopenfilename(initialdir = address_of_filters,title = "Select file",filetypes=(("Filters", "*.txt"),("All files","*.*")))
    filename = root.filename
    if  (type(filename)==str):
        filename_extension = os.path.splitext(filename)[-1]
        base = os.path.basename(filename)
        base_name = os.path.splitext(base)[0]
        if  filename_extension == ".txt":
            do_list_add_filter(self='', name_of_file=base_name)
    root.destroy()

def do_list_add_filter(self, name_of_file):
    """Добавить новый фильтр. Принимает название файла без расширения. Ищет в папке Filters"""
    global filters, filters_number
    if (name_of_file!=''):
        filters_number += 1
        filters[filters_number]=name_of_file

def do_list_clear_filters(self, args):
    """Очистить список фильтров и сбросить их счетчик"""
    global filters, filters_number
    filters={}
    filters_number=0

def do_list_rem_filter(self, number):
    """Удалить последний фильтр. Принимает номер в списке <int>"""
    global filters, filters_number
    del filters[filters_number]
    filters_number -=1

def do_save_filters (self, args):
    """Сохраняет список фильтров в файл spectrograph_filters.pkl в папку загруженного снимка"""
    global filters, path
    with open(path+'spectrograph_filters.pkl','wb') as dir_save_file:
        pickle.dump(filters, dir_save_file)

def do_print_filters(self, args):
    """Выводит словарь фильтров"""
    global filters
    print("Filters: ", filters)

def do_ask_save_file(self, args):
    """Открытие GUI окна выбора файла для сохранения"""
    global plot
    root = Tk()
    root.withdraw()
    root.filename = filedialog.asksaveasfilename(filetypes=(("PNG files only","*.png"),("All files","*.*")))
    file_name= root.filename
    plt.savefig(file_name)

def do_ask_open_file(self, args):
    """Открытие GUI окна выбора файла для открытия"""
    global initdir, basename, filename
    root = Tk()
    root.withdraw()
    root.filename =  filedialog.askopenfilename(initialdir = initdir,title = "Select file",filetypes=(("Data files only", "*.dat"),("PNG files only","*.png"),("All files","*.*")))
    if  (root.filename):
        filename_extension = os.path.splitext(root.filename)[-1]
        directory= os.path.dirname(root.filename)
        basename= os.path.basename(root.filename)
        with open(address_of_last_dir_savefile,'wb') as dir_save_file:
            pickle.dump(directory, dir_save_file)
        initdir=directory
        if  filename_extension == ".png":
            do_image_to_array(self='', name_of_file=root.filename)
        elif filename_extension == ".dat":
            do_data_to_array(self='', name_of_file=root.filename)
        filename = root.filename
        basepathname =os.path.basename(os.path.dirname(filename))
        do_set_parameters(self='',pathname=os.path.dirname(filename), dirname=basepathname)
    root.destroy()

def do_image_to_array(self, name_of_file):
    global array, this_array_has_a_plot
    this_array_has_a_plot= False
    array= plt.imread(name_of_file)

def do_data_to_array(self, name_of_file):
    global array, this_array_has_a_plot
    this_array_has_a_plot= False
    array= np.fromfile(name_of_file, dtype='>i2')
    array= np.reshape(array[4:], (array[1],array[3]))

def get_freq():
    """Функция uз старых файлов Origin"""
    global array
    global grate, array, freq
    image_size=array.shape[1]
    if (grate==300):
        offset=1001
        dispersion=0.12505
    elif (grate==600):
        offset=1011
        dispersion=0.05918
    else:
        print("Wrong grate")
        return None
    freq_array=[round((i-1-image_size+offset)*dispersion+freq) for i in range(1,image_size+1)]
    return freq_array

def get_angles():
    """Функция из старых файлов Origin"""
    global array
    image_size=array.shape[0]
    angle=[round(-0.0175*(i-1))+11 for i in range(1,image_size+1)]
    return angle

def find_nearest(array, value):
    """Index of nearest"""
    array= np.asarray(array)
    idx =(np.abs(array-value)).argmin()
    return idx

def do_set_freq_step(self,args):
    """Выбор шага оси графика: set... <int>"""
    global freq_step
    freq_step=args
def do_set_angle_step(self,args):
    """Выбор шага оси графика: set... <int>"""
    global angle_step
    angle_step=args

def do_set_rotate(self,args):
    """Включение режима поворота на 180"""
    global rot180
    if (args):
        rot180=True
    else :
        rot180=False

def do_plot (self, args): #args активирует режим вывода в файл, без графика
    """Открывает окно с графиком и текущими настройками в неблокирующем режиме"""
    global freq_from, freq_to, this_array_has_a_plot, plot, graph_title, rot180, freq_step, angle_step, array, scale, grate, filters, filters_number
    #Блокировка перепостроения графика
    if (this_array_has_a_plot):
        print ("You have already made plot for this file. Please open another file (or the same again) and call plot(). Duplicate is prohibited")
    else :
        this_array_has_a_plot= True
        #Подготовка массива к применению фильтров
        (bd_mult, bd_single)= read_bd_map(address_of_bd_map)
        apply_bd_map(array, bd_mult, bd_single)
        if (rot180):
            do_rotate(self='', args=2)
        if (array[1,1]<=1):
            array-=array[1,1] #вычитание фона из изображений
        else:
            array-=array[1,1] #вычитание из импортированных dat
        array[array<0] = 0
        freq_array=get_freq()
        angle_array=get_angles()

        #Применение фильтров
        image_size=array.shape[1]
        array_factor= np.ones(image_size)
        do_list_add_filter("", name_of_file="Camera")
        if   (grate==300):
            do_list_add_filter("", name_of_file="300")
        elif (grate==600):
            do_list_add_filter("", name_of_file="600")
        elif (grate==900):
            do_list_add_filter("", name_of_file="900")

        for key, value in filters.items():
            filter_array=np.loadtxt(address_of_filters+'/'+value+'.txt')
            x=(filter_array[:,0]).transpose()
            y=(filter_array[:,1]).transpose()
            filter_function= interpolate.interp1d(x,y, fill_value="extrapolate")
            filter_vector_function= np.vectorize(filter_function)
            array_factor*=filter_vector_function(freq_array)
        do_list_rem_filter('', len(filters))
        do_list_rem_filter('', len(filters))
        filters_number= filters_number-2
        array_factor_reciprocal=np.reciprocal(array_factor)
        array_factor_rec_diag=np.diag(array_factor_reciprocal)
        array= array @ array_factor_rec_diag

        MAX=array.max()
        array *= 1.0/MAX
        if (scale=='log'):
            array= np.log(array)

        plot = sns.heatmap(array, cmap="nipy_spectral", cbar_kws={'label':'Относительная интенсивность'})
        plot.set_ylabel('Угол, мрад')
        plot.set_xlabel('Длина волны, нм')
        plot.set_title(graph_title)

        #Изменение меток на осях
        min_freq=freq_step*ceil(freq_array[0]/freq_step)
        max_freq=freq_step*floor(freq_array[-1]/freq_step)
        new_label=range(min_freq,max_freq+freq_step,freq_step)
        new_tick= [find_nearest(freq_array,new_label[i]) for i in range (0, len(new_label))]
        plt.xticks(ticks=new_tick, labels=new_label, rotation=0)
        min_angle=angle_step*ceil(angle_array[-1]/angle_step)
        max_angle=angle_step*floor(angle_array[0]/angle_step)
        new_label=range(min_angle,max_angle+angle_step,angle_step)
        new_tick= [find_nearest(angle_array,new_label[i]) for i in range (0, len(new_label))]
        plt.yticks(ticks=new_tick, labels=new_label)
        if (freq_from and freq_to):
            x_from=find_nearest(freq_array,freq_from)
            x_to=find_nearest(freq_array,freq_to)
            plt.xlim(x_from, x_to)

        if (args!='no_plot'):
            plt.ion()
            plt.show()
            plt.tight_layout()
        else:
            np.savetxt(address_of_save_fig+'/'+basename, array, delimiter=",")
            

def do_exit (self, args):
    """Выход из работы. Альтернатива CTRL+c затем ENTER"""
    print (REDCOLOR +  "\nЗавершение работы..."+ NORMALCOLOR)
    sys.exit()

def do_print_array (self,args):
    """Выводит массив в текстовом режиме"""
    print (array)

def do_set_freq (self, args):
    """Изменят частоту на указанную set_freq <int>"""
    global freq
    freq=int(args)
    print(PINKCOLOR+"Frequency "+NORMALCOLOR +str(freq))
def do_set_grate (self, args):
    """Изменят решетку на указанную set_grate<int>"""
    global grate
    grate=int(args)
    print(PINKCOLOR+"Grating "+NORMALCOLOR +str(grate))
def do_set_scale(self, args):
    """Изменят шкалу на указанную set_scale <lin> or <log>"""
    global scale
    scale=args
    print(PINKCOLOR+"Scale "+NORMALCOLOR +scale)
def do_set_title(self, args):
    """Изменят заголовок на указанный set_title <str>"""
    global graph_title
    graph_title=args
    print(PINKCOLOR+"Title "+NORMALCOLOR +graph_title)
def do_print_parameters(self, args):
    """Вывод параметров в порядке аргументов у функции set"""
    global freq, grate, rot180, scale, graph_title, path
    param_turple=(freq, grate, rot180, scale, graph_title)
    print (param_turple)

def do_save_parameters (self, args):
    """Сохраняет параметры осей и заголовка в файл spectrograph_parameters.pkl в папку загруженного снимка"""
    global freq, grate, rot180, scale, graph_title, path
    param_turple=(freq, grate, rot180, scale, graph_title)
    with open(path+'spectrograph_parameters.pkl','wb') as dir_save_file:
        pickle.dump(param_turple, dir_save_file)

def do_set_parameters (self, pathname="", frequency=0, grating=0, dirname=False, rotate=False, scaletype=False, title=None):
    global freq, grate, rot180, scale, graph_title, path, filters, filters_number
    print (PINKCOLOR+"Введенный/"+GREENCOLOR+'сохраненный'+REDCOLOR +"/по умолчанию" + NORMALCOLOR +" параметр:")

    path=pathname
    if (pathname!=''):
        path= path+"/"
    if os.path.isfile(path+'spectrograph_parameters.pkl'):
        with open(path+'spectrograph_parameters.pkl','rb') as dir_save_file:
           param_turple= pickle.load(dir_save_file)
    else:
        param_turple=(None,None,None,None,None)
    if os.path.isfile(path+'spectrograph_filters.pkl'):
        with open(path+'spectrograph_filters.pkl','rb') as dir_save_file:
           filters= pickle.load(dir_save_file)
           filters_number=len(filters)
    #---begin freq
    if (frequency):
        freq=frequency
        print(PINKCOLOR+"Frequency "+NORMALCOLOR +str(freq))
    elif (param_turple[0]):
        freq=param_turple[0]
        print(GREENCOLOR+"Frequency "+NORMALCOLOR +str(freq))
    else:
        freq=800
        print(REDCOLOR+"Frequency "+NORMALCOLOR +str(freq))
    #---begin grate
    if (grating):
        grate=grating
        print(PINKCOLOR+"Grating "+NORMALCOLOR +str(grate))
    elif (param_turple[1]):
        grate=param_turple[1]
        print(GREENCOLOR+"Grating "+NORMALCOLOR +str(grate))
    else:
        grate=300
        print(REDCOLOR+"Grating "+NORMALCOLOR +str(grate))
    #---begin rot
    if (rotate):
        rot180=True
        print(PINKCOLOR+"Rotate "+NORMALCOLOR +'True')
    elif (param_turple[2]):
        rot180=True
        print(GREENCOLOR+"Rotate "+NORMALCOLOR +'True')
    else:
        rot180=True
        print(REDCOLOR+"Rotate "+NORMALCOLOR +'True')
    #---begin scale
    if (scaletype):
        scale=scaletype
        print(PINKCOLOR+"Scale "+NORMALCOLOR +scale)
    elif (param_turple[3]):
        scale=param_turple[3]
        print(GREENCOLOR+"Scale "+NORMALCOLOR +scale)
    else:
        scale='lin'
        print(REDCOLOR+"Scale "+NORMALCOLOR +scale)
    #---begin title
    if (title):
        graph_title=title
        print(PINKCOLOR+"Title "+NORMALCOLOR+ graph_title)
    elif (dirname and dirname!='..'):
        graph_title=dirname
        print(PINKCOLOR+"Title (dir) "+NORMALCOLOR +graph_title)
    elif (param_turple[4]):
        graph_title=param_turple[4]
        print(GREENCOLOR+"Title "+NORMALCOLOR+graph_title )
    else:
        graph_title= "Частотно-угловой спектр филамента"
        print(REDCOLOR+"Title "+NORMALCOLOR+ grath_title)
    #---begin filters
    do_print_filters(self='', args='')
