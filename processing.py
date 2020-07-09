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

x_corner = 0
y_corner = 0
x_width = 1920
y_height = 1200

energy_min = 0
energy_max = 0


def set_energy_limits(enfrom, ento):
    global energy_min, energy_max
    energy_min = enfrom
    energy_max = ento


def do_processing_plot(self, mode):

    def FAS_3D_en():
        global_filename = PB.global_filename

        freq_class = PB.x_axis_frequency()
        freq_array = freq_class.get_freq_unrounded()
        angle_array = PB.get_angles_unrounded()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pathname = os.path.dirname(global_filename)
        filename_extension = os.path.splitext(global_filename)[-1]
        if (pathname):
            for file in os.listdir(pathname):
                if file.endswith(filename_extension):
                    energy = float(file[:file.find("_")])
                    if (energy_max >= energy and energy >= energy_min):
                        if file.endswith(".png"):
                            PB.do_image_to_array('', pathname+"/"+file)
                        elif file.endswith(".dat"):
                            PB.do_data_to_array('', pathname+"/"+file)
                        PB.preprocessing_plot()
                        coordinates = peak_local_max(
                            ndi.uniform_filter(PB.array, 20), min_distance=40)
                        center_of_mass_y = int(ndi.measurements.center_of_mass(PB.array)[0])
                        for i, j in zip(coordinates[:, 0], coordinates[:, 1]):
                            # 820 нм - длина накачки, 0.2 из 1.0 по интенсивности отсечет шум
                            if PB.array[i, j] > 0.2 and freq_array[j] > 820:
                                ax.scatter3D([freq_array[j]], [
                                             energy], angle_array[i-center_of_mass_y+600], c=[PB.array[i, j]], vmin=0.2, vmax=1, cmap='Blues')
        ax.set_xlabel('Длина волны, нм')
        ax.set_ylabel('Энергия, мДж')
        ax.set_zlabel('Угол, мрад')
        plt.show(block=False)

    def FAS_3D_dist():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        filename_extension = ".dat"

        for pathname in ['F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_360cm_830nm_ns10_mode_f900nm_ns12P_gain20',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_150cm_815nm_ns10_mode_f850nm_ns12P_gain10',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_190cm_815nm_ns10_mode_f850nm_ns12_6_gain10',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_210cm_815nm_ns10_mode_f850nm_ns12_6_gain10',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_230cm_815nm_ns10_mode_f850nm_ns12_6_gain10',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_250cm_815nm_ns10_mode_f850nm_ns12P_10old_gain10',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_260cm_815nm_ns10_mode_f850nm_ns12_10old_gain10_true',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_270cm_815nm_ns10_mode_f850nm_ns12_10old_gain10_real_true',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_280cm_815nm_ns10_mode_f850nm_ns12_10old_gain10_real_true',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_290cm_815nm_ns10_mode_f850nm_ns12P_10old_gain10',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_300cm_815nm_ns10_mode_f850nm_ns12_6_gain20',
                         'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_310cm_815nm_ns10_mode_f850nm_ns12_gain20',
                         'F:/Филаментация/Спектры/2019/август/1_matched(using2augcalibration)/F3m_AM_no_lambda_310cm_810nm_ns10_mode_f850nm_ns12_gain20',
                         'F:/Филаментация/Спектры/2019/август/1_matched(using2augcalibration)/F3m_AM_no_lambda_320cm_810nm_ns10_mode_f850nm_ns12_gain20_true',
                         'F:/Филаментация/Спектры/2019/август/1_matched(using2augcalibration)/F3m_AM_no_lambda_330cm_810nm_ns10_mode_f850nm_ns12_gain20_true',
                         'F:/Филаментация/Спектры/2019/август/1_matched(using2augcalibration)/F3m_AM_no_lambda_340cm_810nm_ns10_mode_f850nm_ns12P_gain20',
                         'F:/Филаментация/Спектры/2019/август/1_matched(using2augcalibration)/F3m_AM_no_lambda_350cm_810nm_ns10_mode_f850nm_ns12P_gain20']:
            PB.do_ask_open_file('', reopen_without_asking_anything=True,
                                this_filename=pathname+'/'+os.listdir(pathname)[1])
            PB.do_set_scale('', 'lin')
            for file in os.listdir(pathname):
                if file.endswith(filename_extension):
                    energy = float(file[:file.find("_")])
                    if (energy_max >= energy and energy >= energy_min):
                        PB.do_data_to_array('', pathname+"/"+file)
                        PB.preprocessing_plot()

                        freq_class = PB.x_axis_frequency()
                        freq_array = freq_class.get_freq_unrounded()
                        angle_array = PB.get_angles_unrounded()
                        coordinates = peak_local_max(
                            ndi.uniform_filter(PB.array, 20), min_distance=40)
                        center_of_mass_y = int(ndi.measurements.center_of_mass(PB.array)[0])
                        for i, j in zip(coordinates[:, 0], coordinates[:, 1]):
                            # 820 нм - длина волны накачки, 20% по интенсивности отсечет шум
                            if PB.array[i, j] > 0.2*PB.array.max() and freq_array[j] > 820:
                                # c=0.20, alpha=PB.array[i,j]/3, vmin=0.15, vmax=0.36 ,
                                norm = plt.Normalize(820, 920)
                                ax.scatter3D([freq_array[j]], [int(pathname.split('_')[
                                             5][0:-2])], angle_array[i-center_of_mass_y+600], c=[freq_array[j]], norm=norm, cmap='nipy_spectral')
        ax.set_xlabel('Длина волны, нм')
        ax.set_ylabel('Расстояние до линзы, см')
        ax.set_zlabel('Угол, мрад')
        for angle in range(270, 360, 1):
            ax.view_init(15, angle)
            filename = PB.address_of_save_fig+'/step'+str(angle)+'.png'
            plt.savefig(filename, dpi=100)
        plt.show(block=False)

    def tight_layout():
        plt.tight_layout()

    def save_pkl():
        freq_class = PB.x_axis_frequency()
        freq_array = freq_class.get_freq_unrounded()
        angle_array = PB.get_angles_unrounded()
        print(PB.global_filename)
        data_frame = pd.DataFrame(PB.array, columns=freq_array, index=angle_array)
        data_frame.to_pickle(PB.address_of_save_pkl+'/'+os.path.basename(PB.global_filename)+".bz2")

    def save_data_frame():
        freq_class = PB.x_axis_frequency()
        freq_array = freq_class.get_freq_unrounded()
        angle_array = PB.get_angles_unrounded()
        data_frame = pd.DataFrame(PB.array, columns=freq_array, index=angle_array)
        data_frame.to_csv(PB.address_of_save_df+'/' +
                          os.path.basename(os.path.dirname(PB.global_filename))+"_csv.txt", sep=' ')

    def finding_local_maxima():
        global x_corner, x_width, y_corner, y_height
        freq_class = PB.x_axis_frequency()
        freq = freq_class.get_freq_unrounded()

        if (PB.freq_from and PB.freq_to):  # обрезка изображения
            x_from = freq_class.index(PB.freq_from)
            x_to = freq_class.index(PB.freq_to)
            x_corner = x_from
            x_width = x_to-x_from
        if (PB.angle_from or PB.angle_to):
            y_corner = PB.angle_from
            y_height = (PB.angle_to-PB.angle_from)
        im = PB.array[y_corner:y_corner+y_height, x_corner:x_corner+x_width]
        im = ndi.uniform_filter(im, 20)

        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=40, mode='constant')

        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance=40)

        def display_results():
            nonlocal im, image_max, coordinates
            fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(PB.array[y_corner:y_corner+y_height, x_corner:x_corner +
                                  x_width], cmap="nipy_spectral", aspect='auto')
            ax[0].axis('off')
            ax[0].set_title('Original')

            # ax[1].imshow(image_max, cmap=plt.cm.gray, aspect='auto')
            # ax[1].axis('off')
            # ax[1].set_title('Maximum filter')

            ax[1].imshow(im,  cmap="nipy_spectral", aspect='auto')
            ax[1].autoscale(False)
            for x, y in zip(coordinates[:, 0], coordinates[:, 1]):
                if im[x, y] > 0.2:
                    ax[1].plot(y, x, 'w.')
            ax[1].axis('off')
            ax[1].set_title('Peak local max')

            fig.tight_layout()
            plt.show(block=False)

        display_results()
        return ()

    functions = {
        'find_max': finding_local_maxima,
        'tl': tight_layout,
        'df': save_data_frame,
        'pkl': save_pkl,
        '3Den': FAS_3D_en,
        '3Ddistance': FAS_3D_dist
    }
    selected_function = functions.get(mode)
    if (selected_function):
        selected_function()
    else:
        print("You can call function using argument: ", functions)
