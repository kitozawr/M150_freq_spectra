#!/usr/bin/env python
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PictureBuilder import *
from matcher import make_dictionary, find_kalibr, read_bin_new_Rudnev, read_raw_Mind_Vision
from processing import do_processing_plot, set_energy_limits, set_energy_for_this_folder

matplotlib.use('TkAgg')
sg.theme('Reddit')

figure_w, figure_h = 650, 480
default_kalibr_folder = 'G:/Мой диск/Филаментация/Энергии/октябрь/3/kalibr_no_lambda_ns10tr_8'
if os.path.isdir('G:/Мой диск/Филаментация/Энергии/'):
    default_folder_for_kalibr_folders = 'G:/Мой диск/Филаментация/Энергии/'
else:
    default_folder_for_kalibr_folders = None


# ------------------------------- Beginning of Matplotlib helper code -----------------------


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


def place(elem, size=(None, None)):
    return sg.Column([[elem]], size=size, pad=(0, 0), element_justification='center')


# ------------------------------- Beginning of GUI CODE -------------------------------
# Menu Definition
menu_def = [['&File', ['&Open     Ctrl-O', '&Save       Ctrl-S', 'E&xit']],
            ['&Edit', ['&Angle step', '&Freq step', '---', 'Freq &limits', 'Angle limits'], ],
            ['&Toolbar', ['---', '&Rotate180',
                          '---', 'Print &array', 'Print &parameters', 'Print &filters']],
            ['&Help', '&About', ]]
# define the window layout
tab1_layout = [[sg.Canvas(size=(figure_w, figure_h), key='canvas')],
               [sg.T('kalibr_folder'),
                sg.In(default_text=default_kalibr_folder,
                      key='kalibr_folder'), sg.FolderBrowse(target='kalibr_folder', key='browse_kalibr_folder',
                                                            initial_folder=default_folder_for_kalibr_folders),
                sg.Text('-', key='-energy-', size=(10, 1)), sg.Text('мДж')]]

tab2_layout = [#[sg.Output(size=(88, 10), key='_output_')],
               [sg.Button('Clear', size=(10, 1))],
               [sg.Push(), sg.Text('Frequency', size=(10, 1)), sg.InputText(key='-FREQ-', size=(11, 1)),
                sg.Button('Set frequency', size=(12, 1)), sg.Text('Grating', size=(10, 1)),
                sg.InputText(key='-GRATE-', size=(11, 1)), sg.Button('Set grate', size=(12, 1)), sg.Push()],
               [sg.Push(), sg.Text('Rotate', size=(10, 1)), sg.InputText(key='-ROT-', size=(11, 1)),
                sg.Button('Set rotate', size=(12, 1)), sg.Text('Scale', size=(10, 1)),
                sg.InputText('lin', key='-SCALE-', size=(11, 1)), sg.Button('Set scale', size=(12, 1)), sg.Push()],
               [sg.Push(), sg.Text('Title', size=(10, 1)), sg.InputText(
                   key='-TITLE-', size=(54, 1), focus=True), sg.Button('Set title', size=(12, 1)), sg.Push()],
               [sg.Text('Parameters:', size=(10, 1)), sg.Button('Save parameters', size=(12, 1))],
               [sg.Checkbox('Normalize', default=True, key='-NORM-'),
                sg.Checkbox('Crop image', default=True, key='-PATCH-', size=(8, 1)),
                sg.Checkbox('Translate into Russian', default=True, key='-RUS-'),
                sg.Checkbox('Insert title', default=True, key='-INSERTTITLE-'),
                sg.Text('Сalibration'), sg.Listbox(values=config.sections(), enable_events=True, default_values='02_2022', key='-NEWCALIBRATION-')],
               [sg.Text('Angle shift=', size=(10, 1)), sg.T('0', key='_SHIFT_', size=(11, 1)),
                sg.Slider((-100, 100), key='_SLIDER_', orientation='h', default_value=0,
                          disable_number_display=True, enable_events=True)],
               [sg.Text('Rotate degr=', size=(10, 1)), sg.T('0', key='_ANGLE_', size=(11, 1)),
                sg.Slider(range=(-50, 50), key='_SLIDERV_', orientation='h', default_value=0, tick_interval=0.1,
                          disable_number_display=True, enable_events=True)],
               [sg.Checkbox('Convert angles back to pixels', default=False, key='-PIXELS-')],
               [sg.Text('_' * 89)],
               [sg.Text('Filters:', size=(10, 1)), sg.Button('Add filter'), sg.Button('Delete last'),
                sg.Button('Clear all'), sg.Button('Save filters')]]

tab3_layout = [[sg.Text('Save preview of the folder to ./Output (take a few minutes)')],
               [sg.Button('Folder Preview'), sg.Checkbox(
                   'Save as .csv', default=False, key='-CSV-'), sg.Checkbox('Mode', default=False, key='-MODE-'),
                sg.Checkbox('Energy', default=False, key='-ENERGY-'), sg.Cancel()],
               [sg.ProgressBar(1000, orientation='h', size=(30, 20), key='progbar')],
               [sg.Frame(layout=[[sg.Button('Save .csv to ./Output'), sg.Button('Save .pkl to ./Output')],
                                 [sg.Button('Average of folder')],
                                 [sg.Text('Enregy limits (mJ)'), sg.InputText('3', key='-ENERGYFROM-',
                                                                              size=(4, 1)),
                                  sg.InputText('22', key='-ENERGYTO-', size=(4, 1))],
                                 [sg.Button('Find local max'), sg.Button('Local max 2D (in this folder)'),
                                  sg.Button('Energy contribution')],
                                 [sg.Button('Local max 3D (in this folder)'),
                                  sg.Button('Local max 3D (in all folders)')]],
                         title='Specific processing functions from processing.py:',
                         relief=sg.RELIEF_SUNKEN, tooltip='Use these to set flags')]]

tab4_layout = [[sg.Canvas(size=(figure_w, figure_h), key='canvas-mode')]]

tab5_layout = [[sg.Text('Choose text element:'), sg.Listbox(['title', 'xlabel', 'ylabel', 'xticklabels', 'yticklabels', 'cbartitle', 'cbarticklabels', 'energy'], size=(20, 8), key='-LIST-', enable_events=True, default_values='title')],
                [sg.Text('_' * 89)],
                [sg.Text('Color:', text_color='blue', font='arial 11'), sg.Input(size=(12,1), enable_events=True, k='-COLORINPUT-'), sg.Push(), sg.Text('Font style: ', font='arial 11 underline'), sg.CB('Bold', key='-bold-', change_submits=True),
                sg.CB('Italics', key='-italics-', change_submits=True),
                sg.CB('Underline', key='-underline-', change_submits=True), sg.Push()],
                [sg.Text('Font size:', font='arial 14'), sg.Slider(range=(0, 50), key='_SLIDERF_', size=(50, 16), tick_interval=5, orientation='h', default_value=0, enable_events=True)],
                [sg.Text('Font family:', font='italic'),  sg.Input(size=(12,1), enable_events=True, k='-COMBO-')],
                [sg.Text('Ignore everything above, use default value:'), sg.CB('', key='-DEFAULT-', change_submits=True)],
                [sg.Button('Save style')]]

layout = [[sg.Menu(menu_def, tearoff=True, pad=(200, 1))],
          [sg.TabGroup([[sg.Tab('Spectrum', tab1_layout), sg.Tab('Parameters', tab2_layout),
                         sg.Tab('Data processing', tab3_layout),
                         sg.Tab('Mode', tab4_layout, key='_tab_mode_', visible=False),
                         sg.Tab('Fonts', tab5_layout)]])],
          [sg.Button('Open'), place(sg.Button('Show', bind_return_key=True, visible=False)),
           place(sg.Button('Save', visible=False)), sg.Button('Exit')]]

# create the form and show it without the plot
window = sg.Window('Частотно-угловой спектр', layout, finalize=True, return_keyboard_events=True)
# add the plot to the window
fig_canvas_agg = None
fig_canvas_agg_mode = None
fig = None
print("This is debug window. Re-routing the stdout")

# The GUI Event Loop
kalibr_m = 0
kalibr_c = 0
previous_folder = ''
kalibr_m, kalibr_c = find_kalibr(default_kalibr_folder)
while True:
    event, values = window.read()
    if event == 'Exit' or event is None:
        break  # exit button clicked
    elif event == 'Open' or event == "o:79" or event == 'Open     Ctrl-O':
        global_filename = do_ask_open_file("")
        window['Show'].update(visible=True)
        window['Save'].update(visible=True)
        # print(previous_folder, os.path.dirname(global_filename))
        try:
            path_to_dictionary = os.path.dirname(global_filename) + '/dictionary_of_match.pkl'
            with open(path_to_dictionary, 'rb') as dict_save_file:
                dictionary_of_match = pickle.load(dict_save_file)
        except:
            dictionary_of_match = make_dictionary(os.path.dirname(global_filename))
        previous_folder = os.path.dirname(global_filename)
    elif event == 'Save' or event == "s:83" or event == 'Save       Ctrl-S':
        do_ask_save_file("", fig)
    elif event == 'Show' or event == "p:80":
        do_plot('', (values["-NORM-"], not values["-PATCH-"],
                     values['_SLIDER_'] / 10., values['_SLIDERV_'] / 10.,
                     values['-RUS-'], values['-INSERTTITLE-'],
                     values['-PIXELS-'], values['-NEWCALIBRATION-']))
        fig = plt.gcf()
        if fig_canvas_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(fig_canvas_agg)
        fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)

        index_pb = os.listdir(os.path.dirname(global_filename)).index(os.path.basename(global_filename))
        print("Show file № " + str(index_pb))
        window['-energy-'].update('-')
        try:
            with open(values['kalibr_folder'] + '/kalibr.pkl', 'rb') as kalibr_save_file:
                kalibr_m, kalibr_c = pickle.load(kalibr_save_file)
        except:
            kalibr_m, kalibr_c = find_kalibr(values['kalibr_folder'])
        if dictionary_of_match.get(index_pb):
            pathname_ac = os.path.dirname(global_filename).replace('Спектры', 'Моды')
            if dictionary_of_match.get(index_pb)[0] > 0 and os.path.isfile(
                    pathname_ac + '/' + os.listdir(pathname_ac)[dictionary_of_match.get(index_pb)[0]]):
                window.find_element('_tab_mode_').Update(visible=True)
                data, width, height = read_raw_Mind_Vision(
                    pathname_ac + '/' + os.listdir(pathname_ac)[dictionary_of_match.get(index_pb)[0]])
                fig = plt.figure()
                plt.imshow(data, cmap='jet', aspect='auto')
                fig_mode = plt.gcf()
                if fig_canvas_agg_mode:
                    # ** IMPORTANT ** Clean up previous drawing before drawing again
                    delete_figure_agg(fig_canvas_agg_mode)
                fig_canvas_agg_mode = draw_figure(window['canvas-mode'].TKCanvas, fig_mode)
            else:
                window.find_element('_tab_mode_').Update(visible=False)
            pathname_en = os.path.dirname(global_filename).replace('Спектры', 'Энергии')
            if os.path.exists(pathname_en + '/TestFolder'):
                pathname_en = pathname_en + '/TestFolder'
            if dictionary_of_match.get(index_pb)[1] > 0 and os.path.isfile(
                    pathname_en + '/' + os.listdir(pathname_en)[dictionary_of_match.get(index_pb)[1]]):
                T, dt, wf0, wf1 = read_bin_new_Rudnev(
                    pathname_en + '/' + os.listdir(pathname_en)[dictionary_of_match.get(index_pb)[1]])
                window['-energy-'].update(str(np.amax(wf0) * kalibr_m + kalibr_c))

    elif event == 'Clear':
        window.FindElement('_output_').Update('')
    elif event == 'Print array':
        do_print_array('', '')
    elif event == 'Rotate180':
        do_rotate_image("", 2)
    elif event == 'Save parameters':
        do_save_parameters('', '')
    elif event == 'Set frequency':
        do_set_freq('', values['-FREQ-'])
    elif event == 'Set grate':
        do_set_grate('', values['-GRATE-'])
    elif event == 'Set rotate':
        do_set_rotate('', values['-ROT-'])
    elif event == 'Set scale':
        do_set_scale('', values['-SCALE-'])
    elif event == 'Set title':
        do_set_title('', values['-TITLE-'])
    elif event == 'Print parameters':
        do_print_parameters('', '')
    elif event == 'Angle step':
        do_set_angle_step('', sg.popup_get_text('Выбор шага оси графика: <int>'))
    elif event == 'Freq step':
        do_set_freq_step('', sg.popup_get_text('Выбор шага оси графика: <int>'))
    elif event == 'Freq limits':
        from PictureBuilder import freq_from, freq_to

        input_text = sg.popup_get_text(
            'Выбор пределов построения графика от [нм] до [нм] через пробел. Сброс при вводе 0 0\n' +
            'Текущее значение ' + str(freq_from) + " " + str(freq_to))
        if input_text and input_text.split()[0].isdigit() and input_text.split()[1].isdigit():
            do_set_freq_limits('', input_text)
    elif event == 'Angle limits':
        from PictureBuilder import angle_from, angle_to

        input_text = sg.popup_get_text(
            'Выбор пределов построения графика от[пиксель] до[пиксель] через пробел от 0 до 1199.\n' +
            'Сброс при вводе 0 0. Текущее значение ' + str(angle_from) + " " + str(angle_to))
        if input_text and input_text.split()[0].isdigit() and input_text.split()[1].isdigit():
            do_set_angle_limits('', input_text)
    elif event == 'Print filters':
        do_print_filters('', '')
    elif event == 'Add filter':
        do_ask_add_filter('', '')
    elif event == 'Delete last':
        do_list_pop_filter('', len(filters))
    elif event == 'Clear all':
        do_list_clear_filters('', '')
    elif event == 'Save filters':
        do_save_filters('', '')
    elif event == 'Folder Preview':
        do_folder_preview(window, values["-CSV-"], dictionary_of_match)
    elif event == 'Save .csv to ./Output':
        do_processing_plot(window, mode='df')
    elif event == 'Save .pkl to ./Output':
        do_processing_plot(window, mode='pkl')
    elif event == 'Find local max':
        do_processing_plot(window, mode='find_max')
    elif event == 'Local max 3D (in this folder)':
        set_energy_limits(float(values["-ENERGYFROM-"]), float(values["-ENERGYTO-"]))
        do_processing_plot(window, mode='3Den')
    elif event == 'Local max 2D (in this folder)':
        set_energy_limits(float(values["-ENERGYFROM-"]), float(values["-ENERGYTO-"]))
        set_energy_for_this_folder(dictionary_of_match, kalibr_m, kalibr_c)
        do_processing_plot(window, mode='2D_folder')
    elif event == 'Local max 3D (in all folders)':
        set_energy_limits(float(values["-ENERGYFROM-"]), float(values["-ENERGYTO-"]))
        do_processing_plot(window, mode='3Ddistance')
    elif event == 'Energy contribution':
        set_energy_limits(float(values["-ENERGYFROM-"]), float(values["-ENERGYTO-"]))
        do_processing_plot(window, mode='energy_red')
    elif event == 'Average of folder':
        do_processing_plot(window, mode='Average of folder')
        fig = plt.gcf()
        if fig_canvas_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(fig_canvas_agg)
        fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)
    elif event == '_SLIDER_':
        window['_SHIFT_'].update(values['_SLIDER_'] / 10)
    elif event == '_SLIDERV_':
        window['_ANGLE_'].update(values['_SLIDERV_'] / 10.0)
    elif event == '-LIST-':
        if values['-LIST-'] == ['title']: i = 0
        if values['-LIST-'] == ['xlabel']: i = 1
        if values['-LIST-'] == ['ylabel']: i = 2
        if values['-LIST-'] == ['xticklabels']: i = 3
        if values['-LIST-'] == ['yticklabels']: i = 4
        if values['-LIST-'] == ['cbartitle']: i = 5
        if values['-LIST-'] == ['cbarticklabels']: i = 6
        if values['-LIST-'] == ['energy']: i = 7
        window['-COLORINPUT-'].update(style_list[i].color)
        window['-bold-'].update(style_list[i].bold)
        window['-italics-'].update(style_list[i].italics)
        window['-underline-'].update(style_list[i].underline)
        window['-COMBO-'].update(style_list[i].family)
        window['_SLIDERF_'].update(style_list[i].size)
        window['-DEFAULT-'].update(style_list[i].default)

    elif event == 'Save style':
        if values['-LIST-'] == ['title']: i = 0
        if values['-LIST-'] == ['xlabel']: i = 1
        if values['-LIST-'] == ['ylabel']: i = 2
        if values['-LIST-'] == ['xticklabels']: i = 3
        if values['-LIST-'] == ['yticklabels']: i = 4
        if values['-LIST-'] == ['cbartitle']: i = 5
        if values['-LIST-'] == ['cbarticklabels']: i = 6
        if values['-LIST-'] == ['energy']: i = 7
        style_list[i].color = values['-COLORINPUT-']
        style_list[i].bold = values['-bold-']
        style_list[i].italics = values['-italics-']
        style_list[i].underline = values['-underline-']
        style_list[i].family = values['-COMBO-']
        style_list[i].size = values['_SLIDERF_']
        style_list[i].default = values['-DEFAULT-']
    elif event == 'About':
        window.disappear()
        sg.popup(' ' * 29 + '~About this program~', 'Webpage: https://github.com/kitozawr/M150_freq_spectra',
                 "Help:        ./README.md", 'Author:     Zhidovtsev Nikita', grab_anywhere=True)
        window.reappear()
window.close()
