from tkinter import *
from tkinter import filedialog
import sys
import os
import pickle
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

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
array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
sns.set()
address_of_last_dir_savefile='/home/student/Desktop/spectrograph_last_dir.pkl'
if os.path.isfile(address_of_last_dir_savefile):
    with open(address_of_last_dir_savefile,'rb') as dir_save_file:
        initdir= pickle.load(dir_save_file)
else:
    with open(address_of_last_dir_savefile,'wb') as dir_save_file:
        pickle.dump('/', dir_save_file)

def do_rotate(self, args):
    """Вращает на 90 градусов против часовой"""
    global array
    array=np.rot90(array)

def do_hello(self, args):
    """Просто выводит 'hello world' на экран"""
    print ("hello world")

def do_ask_save_file(self, args):
    """Открытие GUI окна выбора файла для сохранения"""
    global plot
    file_name= filedialog.asksaveasfilename(filetypes=(("PNG files only","*.png"),("All files","*.*")))
    plt.savefig(file_name)

def do_ask_open_file(self, args):
    """Открытие GUI окна выбора файла для открытия"""
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = initdir,title = "Select file",filetypes=(("Data files only", "*.dat"),("PNG files only","*.png"),("All files","*.*")))
    filename_extension = os.path.splitext(root.filename)[-1]
    directory= os.path.dirname(root.filename)
    with open(address_of_last_dir_savefile,'wb') as dir_save_file:
        pickle.dump(directory, dir_save_file)

    if  filename_extension == ".png":
        do_image_to_array(self='', name_of_file=root.filename)
    elif filename_extension == ".dat":
        do_data_to_array(self='', name_of_file=root.filename)
    filename = root.filename
    basepathname =os.path.basename(os.path.dirname(filename))
    do_set_parameters(self='',pathname=os.path.dirname(filename), dirname=basepathname)


def do_image_to_array(self, name_of_file):
    global array
    array= plt.imread(name_of_file)

def do_data_to_array(self, name_of_file):
    global array
    array= np.fromfile(name_of_file, dtype='>i2')
    array= np.reshape(array[4:], (array[1],array[3]))

def do_plot (self, args):
    """Открывает окно с графиком и текущими настройками в неблокирующем режиме"""
    global plot
    plot = sns.heatmap(array, cmap="nipy_spectral")
    plt.ion()
    plt.show()

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
    freq=args
    print(PINKCOLOR+"Frequency "+NORMALCOLOR +str(freq))
def do_set_grate (self, args):
    """Изменят решетку на указанную set_grate<int>"""
    global grate
    grate=args
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

def do_set_parameters (self, pathname="", frequency=0, grating=0, dirname=False, rotate=False, scaletype='lin', title=None):
    global freq, grate, rot180, scale, graph_title, path

    path=pathname
    if (pathname!=''):
        path= path+"/"
    if os.path.isfile(path+'spectrograph_parameters.pkl'):
        with open(path+'spectrograph_parameters.pkl','rb') as dir_save_file:
           param_turple= pickle.load(dir_save_file)
    else:
        param_turple=(None,None,None,None,None)
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
        rot180=False
        print(GREENCOLOR+"Rotate "+NORMALCOLOR +'False')
    #---begin scale
    if (scaletype):
        scale=scaletype
        print(PINKCOLOR+"Scale "+NORMALCOLOR +scale)
    elif (param_turple[3]):
        scale=param_turple[3]
        print(GREENCOLOR+"Scale "+NORMALCOLOR +scale)
    else:
        scale='lin'
        print(GREENCOLOR+"Scale "+NORMALCOLOR +scale)
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
