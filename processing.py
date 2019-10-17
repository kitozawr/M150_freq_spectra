import PictureBuilder.PictureBuilder as PB
import matplotlib.pylab as plt
import matplotlib.patches as patches
x_corner=0
y_corner=1200-790
x_width= 1920
y_height= 5


def do_processing_plot(self, mode):
    def default():
        print("aaaa")

    def draw_rectangle():
        global x_corner, x_width, y_corner, y_height
        rect = patches.Rectangle((x_corner,y_corner),x_width,y_height,linewidth=1,edgecolor='r',facecolor='none')
        PB.plot.add_patch(rect)

    def save_cropped_image():
        global x_corner, x_width, y_corner, y_height
        freq_array=PB.get_freq(rounded=0)
        subarray=PB.array[y_corner:y_corner+y_height,x_corner:x_corner+x_width]
        mean_subarray= subarray.mean(axis=0)
        print (PB.global_basename[:PB.global_basename.find("_")], mean_subarray.max(), freq_array[mean_subarray.argmax()+x_corner])
        #plt.imsave(address_of_save_fig+'/'+global_basename.replace('dat','jpg'),  subarray)
        plt.imsave(PB.address_of_save_fig+'/'+PB.global_basename.replace('dat','png'),  PB.array)
        f= open(PB.address_of_save_fig+'/'+PB.global_basename.replace('dat','txt'), "a")
        #np.savetxt(f, mean_subarray, fmt='%1.4f')
        f.close()

    functions = {
                    'draw': draw_rectangle,
                    'save_crop': save_cropped_image,
                    'default': default
                }
    selected_function = functions.get(mode)
    if (selected_function):
        selected_function()
    else:
        print ("You can call function using argument: ", functions)
