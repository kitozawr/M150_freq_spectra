from tkinter import *
from tkinter import filedialog
import sys
import os
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
sns.set()
plot = sns.heatmap(array)

def do_hello(self, args):
    """Просто выводит 'hello world' на экран"""
    print ("hello world")

def do_ask_open_file(self, initdir="/"):
    """Открытие GUI окна выбора файла для открытия"""
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = initdir,title = "Select file",filetypes=(("Data files only", "*.dat"),("PNG files only","*.png"),("All files","*.*")))
    filename_extension = os.path.splitext(root.filename)[-1]

    if  filename_extension == ".png":
        do_image_to_array(self='', name_of_file=root.filename)
    elif filename_extension == ".dat":
        do_data_to_array(self='', name_of_file=root.filename)

def do_image_to_array(self, name_of_file):
    global array
    print (name_of_file)
    array= plt.imread(name_of_file)

def do_data_to_array(self, name_of_file):
    global array
    array= np.fromfile(name_of_file, dtype='>i2')
    array= np.reshape(array[4:], (array[1],array[3]))

def do_plot (self, args):
    """Открывает окно с графиком по текущим настройкам в неблокирующем режиме"""
    global plot
    plot = sns.heatmap(array, cmap="nipy_spectral")
    plt.ion()
    plt.show()

def do_exit (self, args):
    """Выход из работы. Альтернатив CTRL+c затем ENTER"""
    sys.exit()

def do_echo (self, args):
    print(args)

def do_print_array (self,args):
    print (array)
