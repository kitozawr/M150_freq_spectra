from tkinter import *
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
sns.set()
plot = sns.heatmap(array)

def do_hello(self, args):
    """Просто выводит 'hello world' на экран"""
    print ("hello world")

def do_ask_open_file(self, initdir="/", filetype="*.jpg;*.dat"):
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = initdir,title = "Select file",filetypes = (("Allowed types",filetype),("all files","*.*")))
    return (root.filename)

def do_image_to_array(self, name_of_file):
    global array
    print (name_of_file)
    array= plt.imread(name_of_file)

def do_data_to_array(self, name_of_file):
    global array
    array= np.fromfile(name_of_file, dtype='>i2')

def do_plot (self, args):
    global plot
    plot = sns.heatmap(array)
    plt.show()

def do_print_array (self,args):
    print (array)
