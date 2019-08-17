#создание массива

def do_hello(self, args):
    """Просто выводит 'hello world' на экран"""
    print ("hello world")

def do_open_file(initdir="/", filetype="*.jpg;*.dat")
    from tkinter import filedialog
    from tkinter import *
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = initdir,title = "Select file",filetypes = (("Allowed types",filetype),("all files","*.*")))
    return (root.filename)
