import PictureBuilder as PB
import matplotlib.pylab as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import matplotlib.cm as cmx
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage as ndi
import os
import pandas as pd
import PySimpleGUI as sg
from mpl_toolkits.mplot3d import Axes3D

x_corner=0
y_corner=0
x_width= 1920
y_height= 1200

def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print('used button  : ', eclick.button)

def toggle_selector(event):
    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def do_processing_plot(self, mode):
    def default():
        #set_parameters(X=0, W=1920, Y=375, H=80)
        finding_local_maxima()
        #fig = plt.figure(dpi=100, tight_layout=True, frameon=False, figsize=(1920/100.,1200/100.))
        #fig.figimage(PB.array, cmap="nipy_spectral")
        #fig.text(0,0,PB.global_basename[:PB.global_basename.find("_")], fontsize=100, backgroundcolor='white', alpha=0.5)
        #plt.savefig(PB.address_of_save_fig+'/'+PB.global_basename.replace('dat','png'))
        #plt.close(fig)
    def FAS_3D_en():
        global_filename=PB.global_filename
        global_basename=PB.global_basename
        array=PB.array
        address_of_save_fig=PB.address_of_save_fig

        freq_class= PB.x_axis_frequency()
        freq_array=freq_class.get_freq_unrounded()
        angle_array=PB.get_angles_unrounded()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pathname=os.path.dirname(global_filename)
        filename_extension = os.path.splitext(global_filename)[-1]
        if (pathname):
            for file in os.listdir(pathname):
                if file.endswith(filename_extension) and float(file[:file.find("_")])>3:
                    global_basename= file
                    if file.endswith(".png"):
                        PB.do_image_to_array('', pathname+"/"+file)
                    elif file.endswith(".dat"):
                        PB.do_data_to_array('', pathname+"/"+file)
                    PB.preprocessing_plot()
                    x,y = finding_local_maxima()
                    for i,j in zip(x,y):
                        if  PB.array[i,j]>0.2 and freq_array[j]>820:
                            ax.scatter3D([freq_array[j]],[float(global_basename[:global_basename.find("_")])], angle_array[i], c=[PB.array[i,j]], vmin=0.2, vmax=1 , cmap='Blues')

        ax.set_xlabel('Frequency, nm')
        ax.set_ylabel('Energy, mJ')
        ax.set_zlabel('Angle, mrad')

        plt.show(block=False)

    def tight_layout():
        plt.tight_layout()

    def save_pkl():
        freq_class= PB.x_axis_frequency()
        freq_array=freq_class.get_freq_unrounded()
        angle_array=PB.get_angles_unrounded()
        data_frame= pd.DataFrame(PB.array, columns= freq_array, index= angle_array)
        data_frame.to_pickle(PB.address_of_save_pkl+'/'+os.path.basename(os.path.dirname(PB.global_filename))+".bz2")

    def save_data_frame():
        freq_class= PB.x_axis_frequency()
        freq_array=freq_class.get_freq_unrounded()
        angle_array=PB.get_angles_unrounded()
        data_frame= pd.DataFrame(PB.array, columns= freq_array, index= angle_array)
        data_frame.to_csv(PB.address_of_save_df+'/'+os.path.basename(os.path.dirname(PB.global_filename))+"_csv.txt", sep=' ')

    def draw_rectangle():
        global x_corner, x_width, y_corners, y_height
        rect = patches.Rectangle((x_corner,y_corner),x_width,y_height,linewidth=1,edgecolor='r',facecolor='none')
        PB.plot.add_patch(rect)

    def save_cropped_image():
        global x_corner, x_width, y_corner, y_height
        freq_class= PB.x_axis_frequency()
        freq_array=freq_class.get_freq_unrounded()
        subarray=PB.array[y_corner:y_corner+y_height,x_corner:x_corner+x_width]
        mean_subarray= subarray.mean(axis=0)
        print (PB.global_basename[:PB.global_basename.find("_")], mean_subarray.max(), freq_array[mean_subarray.argmax()+x_corner])
        #plt.imsave(address_of_save_fig+'/'+global_basename.replace('dat','jpg'),  subarray)
        plt.imsave(PB.address_of_save_fig+'/'+PB.global_basename.replace('dat','png'),  PB.array)

        ###Вывод среза в файл
        f= open(PB.address_of_save_fig+'/'+PB.global_basename.replace('dat','txt'), "a")
        #np.savetxt(f, mean_subarray, fmt='%1.4f')
        f.close()

    def draw_a_rectangle_with_a_mouse():
        toggle_selector.RS = RectangleSelector(PB.plot, onselect, drawtype='line')

    def set_parameters(X=None, W=None, Y=None, H=None):
        global x_corner, x_width, y_corner, y_height
        def set_single_par(X, name_of_X, value_of_x=None):
            print (name_of_X, '= ', X)
            if (value_of_x is None):
                buffer=input()
            else:
                buffer= value_of_x
            if (buffer):
                X= int(buffer)
            return X

        x_corner= set_single_par(x_corner, 'x_corner', X)
        y_corner= set_single_par(y_corner, 'y_corner', Y)
        x_width= set_single_par(x_width, 'x_width', W)
        y_height= set_single_par(y_height, 'y_height', H)

    def finding_local_maxima():
        global x_corner, x_width, y_corner, y_height
        freq_class= PB.x_axis_frequency()
        freq=freq_class.get_freq_unrounded()

        if (PB.freq_from and PB.freq_to): #   обрезка изображения
            x_from=freq_class.index(PB.freq_from)
            x_to=freq_class.index(PB.freq_to)
            x_corner=x_from
            x_width=x_to-x_from
        if (PB.angle_from or PB.angle_to):
            y_corner= PB.angle_from
            y_height= (PB.angle_to-PB.angle_from)
        im = PB.array[y_corner:y_corner+y_height,x_corner:x_corner+x_width]
        im = ndi.uniform_filter(im,20)

        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=40, mode='constant')

        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance=40)

        def display_results():
            nonlocal im, image_max, coordinates
            fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(PB.array[y_corner:y_corner+y_height,x_corner:x_corner+x_width], cmap=plt.cm.gray, aspect='auto')
            ax[0].axis('off')
            ax[0].set_title('Original')

            ax[1].imshow(image_max, cmap=plt.cm.gray, aspect='auto')
            ax[1].axis('off')
            ax[1].set_title('Maximum filter')

            ax[2].imshow(im, cmap=plt.cm.gray, aspect='auto')
            ax[2].autoscale(False)
            for x, y in zip(coordinates[:, 0],coordinates[:, 1]):
                if im[x,y]>0.2:
                    ax[2].plot(y, x, 'r.')
            ax[2].axis('off')
            ax[2].set_title('Peak local max')

            fig.tight_layout()
            plt.show(block=False)


        #f= open(PB.address_of_save_fig+'/'+PB.global_basename.replace('dat','txt'), "a")
        f= open(PB.address_of_save_fig+'/'+os.path.basename(os.path.dirname(PB.global_filename))+".txt", "a")
        np.savetxt(f, [[freq[i] for i in (coordinates[:, 1]) if freq[i]>800]], fmt='%1.4f')
        f.close()

        f= open(PB.address_of_save_fig+'/'+os.path.basename(os.path.dirname(PB.global_filename))+"_energy.txt", "a")
        f.write(PB.global_basename[:PB.global_basename.find("_")]+"\n")
        f.close()
        #display_results()
        return (coordinates[:, 0],coordinates[:, 1])



    functions = {
                    'prob': draw_a_rectangle_with_a_mouse,
                    'draw': draw_rectangle,
                    'set_par': set_parameters,
                    'save_crop': save_cropped_image,
                    'find_max': finding_local_maxima,
                    'tl': tight_layout,
                    'df': save_data_frame,
                    'pkl': save_pkl,
                    '3Den': FAS_3D_en,
                    'default': default
                }
    selected_function = functions.get(mode)
    if (selected_function):
        selected_function()
    else:
        print ("You can call function using argument: ", functions)
