import PictureBuilder as PB
from matcher import read_bin_new_Rudnev
import matplotlib.pylab as plt
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import os

x_corner = 0
y_corner = 0
x_width = 1920
y_height = 1200

energy_min = 0
energy_max = 0


window_copy = None
dictionary= 0
m = 1
cnst = 0

def set_energy_limits(enfrom, ento):
    global energy_min, energy_max
    energy_min = enfrom
    energy_max = ento

def set_energy_for_this_folder(a,b,c):
    global dictionary, m, cnst
    dictionary = a
    m = b
    cnst = c

def do_processing_plot(window, mode):
    def energy_full_vs_energy_in_red():
        global_filename = PB.global_filename

        freq_class = PB.x_axis_frequency()
        freq_array = freq_class.get_freq_unrounded()
        angle_array = PB.get_angles_unrounded()

        pathname = os.path.dirname(global_filename)
        filename_extension = os.path.splitext(global_filename)[-1]
        # решаем в 2 захода: сначала среднее по поддиапазонам 1мдж,
        # потом считаем отклонение от среднего
        max_energy_mj = 30
        energy_index = np.arange(1, max_energy_mj + 1)
        number_of_samples = np.zeros(max_energy_mj)
        sum_of_energy = np.zeros(max_energy_mj)
        mean_of_energy = np.zeros(max_energy_mj)
        sd_of_energy = np.zeros(max_energy_mj)
        sum_of_energy_red = np.zeros(max_energy_mj)
        mean_of_energy_red = np.zeros(max_energy_mj)
        sd_of_energy_red = np.zeros(max_energy_mj)

        if (pathname):
            for file in os.listdir(pathname):
                if file.endswith(filename_extension):
                    energy = float(file[:file.find("_")])
                    if (energy > 1):
                        if file.endswith(".png"):
                            PB.do_image_to_array('', pathname + "/" + file)
                        elif file.endswith(".dat"):
                            PB.do_data_to_array('', pathname + "/" + file)
                            PB.preprocessing_plot()
                            subarray = PB.array[PB.angle_from:PB.angle_to, :]
                            number_of_samples[round(energy)] = number_of_samples[round(energy)] + 1
                            sum_of_energy[round(energy)] = sum_of_energy[round(
                                energy)] + subarray.sum()
                            x_from = freq_class.index(840)
                            subarray = PB.array[PB.angle_from:PB.angle_to, x_from:]
                            sum_of_energy_red[round(energy)] = sum_of_energy_red[round(
                                energy)] + subarray.sum()
        mean_of_energy = np.divide(sum_of_energy, number_of_samples)
        mean_of_energy_red = np.divide(sum_of_energy_red, number_of_samples)

        if (pathname):
            for file in os.listdir(pathname):
                if file.endswith(filename_extension):
                    energy = float(file[:file.find("_")])
                    if (energy > 1):
                        if file.endswith(".png"):
                            PB.do_image_to_array('', pathname + "/" + file)
                        elif file.endswith(".dat"):
                            PB.do_data_to_array('', pathname + "/" + file)
                            PB.preprocessing_plot()
                            subarray = PB.array[PB.angle_from:PB.angle_to, :]
                            sd_of_energy[round(energy)] = sd_of_energy[round(
                                energy)] + np.power(subarray.sum() - mean_of_energy[round(energy)], 2)
                            x_from = freq_class.index(820)
                            subarray = PB.array[PB.angle_from:PB.angle_to, x_from:]
                            sd_of_energy_red[round(energy)] = sd_of_energy_red[round(
                                energy)] + np.power(subarray.sum() - mean_of_energy_red[round(energy)], 2)
        # sd = sqrt(sum[(xi-<x>)^2]/(n-1))
        number_of_samples_minus_1 = np.zeros(max_energy_mj)
        number_of_samples_minus_1[number_of_samples >
                                  1] = number_of_samples[number_of_samples > 1] - 1
        number_of_samples_minus_1[number_of_samples <=
                                  1] = number_of_samples[number_of_samples <= 1]
        sd_of_energy = np.sqrt(np.divide(sd_of_energy, number_of_samples_minus_1))
        sd_of_energy_red = np.sqrt(np.divide(sd_of_energy_red, number_of_samples_minus_1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        '''
        ax.bar(energy_index, mean_of_energy, yerr=sd_of_energy,
               error_kw={'ecolor': '0.1', 'capsize': 6}, label='Full')
        ax.bar(energy_index, mean_of_energy_red,  yerr=sd_of_energy_red,
               error_kw={'ecolor': 'tab:purple', 'capsize': 6}, label='Red')
        '''
        energy_ratio = np.zeros(max_energy_mj)
        energy_ratio[number_of_samples > 1] = np.divide(
            mean_of_energy_red, mean_of_energy)[number_of_samples > 1]
        div = np.divide(energy_ratio * sd_of_energy, mean_of_energy)
        div_red = np.divide(energy_ratio * sd_of_energy_red, mean_of_energy_red)
        # print(div)
        # print("!\n")
        # print(div_red)
        # print("!\n")
        # print(np.divide(sd_of_energy, mean_of_energy))
        # print("!\n")
        energy_ratio_error = (np.sqrt(np.power(div, 2) +
                                      np.power(div_red, 2)))
        # print(energy_ratio_error)
        ax.errorbar(energy_index[number_of_samples > 1],
                    energy_ratio[number_of_samples > 1],
                    yerr=energy_ratio_error[number_of_samples > 1],
                    fmt='o', capsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Энергия в импульсе, мДж')
        ax.set_ylabel('Энергия на оси')
        plt.title('Энерговклад в красное крыло')
        plt.legend(loc=2)
        fig.show()

    def average():
        list_of_maximum = np.array([])
        global_filename = PB.global_filename
        pathname = os.path.dirname(global_filename)
        filename_extension = os.path.splitext(global_filename)[-1]
        PB_sum = np.zeros_like(PB.array)
        if (pathname):
            for file in os.listdir(pathname):
                if file.endswith(filename_extension):
                    if file.endswith(".png"):
                        PB.do_image_to_array('', pathname + "/" + file)
                    elif file.endswith(".dat"):
                        PB.do_data_to_array('', pathname + "/" + file)
                    PB.preprocessing_plot()
                    PB_sum = PB_sum + PB.array
                    #list_of_maximum = np.append(list_of_maximum, np.array(finding_local_maxima(show=False)))
                    #print(finding_local_maxima(show=False))
        #print(list_of_maximum)
        #print(np.mean(list_of_maximum[~np.isnan(list_of_maximum)]))
        #print(np.std(list_of_maximum[~np.isnan(list_of_maximum)]))
        PB.array = PB_sum / PB_sum.max()
        PB.show_plot()

    def FAS_2D_folder():
        global dictionary, m, cnst
        global energy_min, energy_max
        dictionary_of_match = dictionary
        c = cnst
        global_filename = PB.global_filename
        pathname = os.path.dirname(global_filename)
        filename_extension = os.path.splitext(global_filename)[-1]
        pathname_en = os.path.dirname(global_filename).replace('Спектры', 'Энергии')
        if (os.path.exists(pathname_en + '/TestFolder')):
            pathname_en = pathname_en + '/TestFolder'
        freq_class = PB.x_axis_frequency()
        freq_array = freq_class.get_freq_unrounded()
        angle_array = PB.get_angles_unrounded()
        for i, file in enumerate(os.listdir(pathname)):
            event, values = window.read(timeout=0)
            if event == 'Cancel':
                break
            window['progbar'].update_bar(int(i / len(os.listdir(pathname)) * 1000) + 1)
            if (file.endswith(filename_extension) ):#and dictionary_of_match.get(i) and dictionary_of_match.get(i)[1] > 0):
                try:
                    global_basename = file
                    if file.endswith(".png"):
                        PB.do_image_to_array('', pathname + "/" + file)
                    elif file.endswith(".dat"):
                        PB.do_data_to_array('', pathname + "/" + file)
                    PB.preprocessing_plot()
                    T, dt, wf0, wf1 = read_bin_new_Rudnev(
                        pathname_en + '/' + os.listdir(pathname_en)[dictionary_of_match.get(i)[1]])
                    # fig.suptitle(str(round(np.amax(wf0) * m + c, 2)) + ' mJ',
                    #              y=1, ha='right', fontsize=12)
                    # print(np.amax(wf0)*m+c, array[angle_from:angle_to + 1, 0: 1900].mean())
                    from PictureBuilder import array
                    coordinates = peak_local_max(
                        ndi.uniform_filter(array, 20), min_distance=40)
                    center_of_mass_y = int(ndi.measurements.center_of_mass(array)[0])
                    time_pb = (int(file.split("_")[-3]) * 3600 + int(
                        file.split("_")[-2]) * 60 + float(
                        file.split("_")[-1][0:6].replace(",", ".")))
                    #print(array.mean(), time_pb)
                    for i, j in zip(coordinates[:, 0], coordinates[:, 1]):
                        # 820 нм - длина накачки, 0.2 из 1.0 по интенсивности отсечет шум
                        if array[i, j] > 0.02 and freq_array[j] > 820 and i>400 and i<500:
                            if True:#(energy_max >= np.amax(wf0)*m+c and np.amax(wf0)*m+c >= energy_min):
                                #plt.scatter(np.amax(wf0)*m+c, freq_array[j])
                                print(np.amax(wf0)*m+c, time_pb, freq_array[j])
                                #pass
                except:
                    print("An exception #1 occurred")
        #plt.show(block=False)


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
                            PB.do_image_to_array('', pathname + "/" + file)
                        elif file.endswith(".dat"):
                            PB.do_data_to_array('', pathname + "/" + file)
                        PB.preprocessing_plot()
                        coordinates = peak_local_max(
                            ndi.uniform_filter(PB.array, 20), min_distance=40)
                        center_of_mass_y = int(ndi.measurements.center_of_mass(PB.array)[0])
                        for i, j in zip(coordinates[:, 0], coordinates[:, 1]):
                            # 820 нм - длина накачки, 0.2 из 1.0 по интенсивности отсечет шум
                            if PB.array[i, j] > 0.2 and freq_array[j] > 820:
                                ax.scatter3D([freq_array[j]], [
                                    energy], angle_array[i - center_of_mass_y + 600], c=[PB.array[i, j]], vmin=0.2,
                                             vmax=1, cmap='Blues')
        ax.set_xlabel('Длина волны, нм')
        ax.set_ylabel('Энергия, мДж')
        ax.set_zlabel('Угол, мрад')
        plt.show(block=False)

    def FAS_3D_dist():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        filename_extension = ".dat"

        for pathname in [
            'F:/Филаментация/Спектры/2019/август/2_matched/F3m_AM_no_lambda_360cm_830nm_ns10_mode_f900nm_ns12P_gain20',
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
                                this_filename=pathname + '/' + os.listdir(pathname)[1])
            PB.do_set_scale('', 'lin')
            for file in os.listdir(pathname):
                if file.endswith(filename_extension):
                    energy = float(file[:file.find("_")])
                    if (energy_max >= energy and energy >= energy_min):
                        PB.do_data_to_array('', pathname + "/" + file)
                        PB.preprocessing_plot()

                        freq_class = PB.x_axis_frequency()
                        freq_array = freq_class.get_freq_unrounded()
                        angle_array = PB.get_angles_unrounded()
                        coordinates = peak_local_max(
                            ndi.uniform_filter(PB.array, 20), min_distance=40)
                        center_of_mass_y = int(ndi.measurements.center_of_mass(PB.array)[0])
                        for i, j in zip(coordinates[:, 0], coordinates[:, 1]):
                            # 820 нм - длина волны накачки, 20% по интенсивности отсечет шум
                            if PB.array[i, j] > 0.2 * PB.array.max() and freq_array[j] > 820:
                                # c=0.20, alpha=PB.array[i,j]/3, vmin=0.15, vmax=0.36 ,
                                norm = plt.Normalize(820, 920)
                                ax.scatter3D([freq_array[j]], [int(pathname.split('_')[
                                                                       5][0:-2])],
                                             angle_array[i - center_of_mass_y + 600], c=[freq_array[j]], norm=norm,
                                             cmap='nipy_spectral')
        ax.set_xlabel('Длина волны, нм')
        ax.set_ylabel('Расстояние до линзы, см')
        ax.set_zlabel('Угол, мрад')
        for angle in range(270, 360, 1):
            ax.view_init(15, angle)
            filename = PB.address_of_save_fig + '/step' + str(angle) + '.png'
            plt.savefig(filename, dpi=100)
        plt.show(block=False)

    def tight_layout():
        plt.tight_layout()

    def save_pkl():
        data_frame = PB.data_frame
        np.save(PB.address_of_save_pkl + '/' + os.path.basename(PB.global_filename) + ".npy", data_frame)

    def save_data_frame():
        data_frame = PB.data_frame
        np.savetxt(PB.address_of_save_df + '/' +
                          os.path.basename(os.path.dirname(PB.global_filename)) + "_csv.txt", data_frame, delimiter=' ')

    def finding_local_maxima(show=True):
        global x_corner, x_width, y_corner, y_height
        freq_class = PB.x_axis_frequency()
        freq = freq_class.get_freq_unrounded()

        if (PB.freq_from and PB.freq_to):  # обрезка изображения
            x_from = freq_class.index(PB.freq_from)
            x_to = freq_class.index(PB.freq_to)
            x_corner = x_from
            x_width = x_to - x_from
        if (PB.angle_from or PB.angle_to):
            y_corner = PB.angle_from
            y_height = (PB.angle_to - PB.angle_from)
        im = PB.array[y_corner:y_corner + y_height, x_corner:x_corner + x_width]
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

            ax[0].imshow(PB.array[y_corner:y_corner + y_height, x_corner:x_corner +
                                                                         x_width], cmap="nipy_spectral", aspect='auto')
            ax[0].axis('off')
            ax[0].set_title('Original')

            # ax[1].imshow(image_max, cmap=plt.cm.gray, aspect='auto')
            # ax[1].axis('off')
            # ax[1].set_title('Maximum filter')

            ax[1].imshow(im, cmap="nipy_spectral", aspect='auto')
            ax[1].autoscale(False)
            for x, y in zip(coordinates[:, 0], coordinates[:, 1]):
                if im[x, y] > 0.02 and freq[y] > 820 and freq[y] < 900:
                    ax[1].plot(y, x, 'w.')
                    #print(y, freq[y])
            ax[1].axis('off')
            ax[1].set_title('Peak local max')

            fig.tight_layout()
            plt.show(block=False)
            #plt.savefig("out.png")

        if show:
            display_results()
        for x, y in zip(coordinates[:, 0], coordinates[:, 1]):
            if im[x, y] > 0.02 and freq[y] > 820 and freq[y] < 900:
                return(freq[y])


    functions = {
        'find_max': finding_local_maxima,
        'tl': tight_layout,
        'df': save_data_frame,
        'pkl': save_pkl,
        '3Den': FAS_3D_en,
        '3Ddistance': FAS_3D_dist,
        '2D_folder': FAS_2D_folder,
        'energy_red': energy_full_vs_energy_in_red,
        'Average of folder': average
    }
    selected_function = functions.get(mode)
    if (selected_function):
        selected_function()
    else:
        print("You can call function using argument: ", functions)
