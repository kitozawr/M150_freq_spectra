import PictureBuilder.PictureBuilder as PB
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import RectangleSelector

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
    x_corner=0
    y_corner=1200-790
    x_width= 1920
    y_height= 5

    def default():
        print("Fine")

    def draw_rectangle():
        nonlocal x_corner, x_width, y_corner, y_height
        rect = patches.Rectangle((x_corner,y_corner),x_width,y_height,linewidth=1,edgecolor='r',facecolor='none')
        PB.plot.add_patch(rect)

    def save_cropped_image():
        nonlocal x_corner, x_width, y_corner, y_height
        freq_array=PB.get_freq(rounded=0)
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

    def set_parameters():
        nonlocal x_corner, x_width, y_corner, y_height
        print ("x: ", x_corner)
        print ("y: ", y_corner)
        print ("width: ", x_width)
        print ("height: ", y_height)


    functions = {
                    'prob': draw_a_rectangle_with_a_mouse,
                    'draw': draw_rectangle,
                    'save_crop': save_cropped_image,
                    'default': default
                }
    selected_function = functions.get(mode)
    if (selected_function):
        selected_function()
    else:
        print ("You can call function using argument: ", functions)
