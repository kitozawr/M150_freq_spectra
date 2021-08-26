#!/usr/bin/env python
from PictureBuilder import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')
sg.theme('Reddit')

figure_w, figure_h = 650, 480
# ------------------------------- Beginning of Matplotlib helper code -----------------------


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


# ------------------------------- Beginning of GUI CODE -------------------------------
# Menu Definition
menu_def = [['&File', ['&Open     Ctrl-O', '&Save       Ctrl-S', 'E&xit']],
            ['&Edit', ['&Angle step', '&Freq step', '---', 'Freq &limits', 'Angle limits'], ],
            ['&Toolbar', ['---', '&Rotate180',
                          '---', 'Print &array', 'Print &parameters', 'Print &filters']],
            ['&Help', '&About...'], ]
# define the window layout
tab1_layout = [[sg.Canvas(size=(figure_w, figure_h), key='canvas')]]

tab2_layout = [[sg.Output(size=(88, 10))],
               [sg.Text('Parameters:',  size=(10, 1)), sg.Button('Save parameters', size=(12, 1))],
               [sg.Text(' ', size=(10, 1)), sg.Text('Frequency', size=(11, 1)),
                sg.InputText(key='-FREQ-'), sg.Button('Set frequency', size=(12, 1))],
               [sg.Text(' ', size=(10, 1)), sg.Text('Grating', size=(11, 1)),
                sg.InputText(key='-GRATE-'), sg.Button('Set grate', size=(12, 1))],
               [sg.Text(' ', size=(10, 1)), sg.Text('Rotate', size=(11, 1)),
                sg.InputText(key='-ROT-'), sg.Button('Set rotate', size=(12, 1))],
               [sg.Text(' ', size=(10, 1)), sg.Text('Scale', size=(11, 1)), sg.InputText(
                   'lin', key='-SCALE-'), sg.Button('Set scale', size=(12, 1))],
               [sg.Text(' ', size=(10, 1)), sg.Text('Title', size=(11, 1)), sg.InputText(
                   key='-TITLE-', focus=True), sg.Button('Set title', size=(12, 1))],
               [sg.Checkbox('Normalize', default=True, key='-NORM-'), sg.Checkbox('Crop image', default=True, key='-PATCH-', size=(8, 1)),
                sg.Checkbox('Translate into Russian', default=True, key='-RUS-'), sg.Checkbox('Insert title', default=True, key='-INSERTTITLE-')],
               [sg.Text('Angle shift=', size=(10, 1)), sg.T('0', key='_SHIFT_',  size=(11, 1)),
                sg.Slider((-100, 100), key='_SLIDER_', orientation='h', default_value=0,
                          disable_number_display=True, enable_events=True)],
               [sg.Text('Rotate degr=', size=(10, 1)), sg.T('0', key='_ANGLE_',  size=(11, 1)),
                sg.Slider(range=(-50, 50), key='_SLIDERV_', orientation='h', default_value=0, tick_interval=0.1, disable_number_display=True, enable_events=True)],
               [sg.Checkbox('Convert angles back to pixels', default=False, key='-PIXELS-')],
               [sg.Text('_' * 89)],
               [sg.Text('Filters:',  size=(10, 1)), sg.Button('Add filter'), sg.Button('Delete last'), sg.Button('Clear all'), sg.Button('Save filters')]]

tab3_layout = [[sg.Text('_' * 89)],
               [sg.Text('Save preview of the folder to ./Output (take a few minutes)')],
               [sg.Button('Folder Preview'), sg.Checkbox(
                   'Save as .csv', default=False, key='-CSV-'), sg.Cancel()],
               [sg.ProgressBar(1000, orientation='h', size=(20, 20), key='progbar')],
               [sg.Text('_' * 89)],
               [sg.Text('Extra options:')],
               [sg.Button('Save .csv to ./Output'), sg.Button('Save .pkl to ./Output')],
               [sg.Text('Enregy limits (mJ)'), sg.InputText('3', key='-ENERGYFROM-',
                                                            size=(4, 1)), sg.InputText('22', key='-ENERGYTO-', size=(4, 1))],
               [sg.Button('Find local max'), sg.Button('Local max 3D (in this folder)'),
                sg.Button('Local max 3D (in all folders)')],
               [sg.Button('Energy contribution')]]

layout = [[sg.Menu(menu_def, tearoff=True, pad=(200, 1))],
          [sg.TabGroup([[sg.Tab('Graph', tab1_layout), sg.Tab(
              'Parameters', tab2_layout), sg.Tab('Data manipulation', tab3_layout)]])],
          [sg.Button('Open'), sg.Button('Show', bind_return_key=True), sg.Button('Save'), sg.Button('Exit')]]

# create the form and show it without the plot
window = sg.Window('Частотно-угловой спектр', layout, finalize=True, return_keyboard_events=True)

# add the plot to the window
fig_canvas_agg = None
fig = None
# add user functions

# The GUI Event Loop
while True:
    event, values = window.read()
    if event == 'Exit' or event is None:
        break  # exit button clicked
    elif event == 'Open' or event == "o:79" or event == 'Open     Ctrl-O':
        do_ask_open_file("")
    elif event == 'Save' or event == "s:83" or event == 'Save       Ctrl-S':
        do_ask_save_file("", fig)
    elif event == 'Show' or event == "p:80":
        do_plot('', (values["-NORM-"], not values["-PATCH-"],
                     values['_SLIDER_']/10., values['_SLIDERV_']/10.,
                     values['-RUS-'], values['-INSERTTITLE-'],
                     values['-PIXELS-']))
        fig = plt.gcf()
        if fig_canvas_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(fig_canvas_agg)
        fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)
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
        input_text = sg.popup_get_text('Выбор пределов построения графика от [нм] до [нм] через пробел. Сброс при вводе 0 0\n' +
                                       'Текущее значение ' + str(freq_from)+" "+str(freq_to))
        if (input_text and input_text.split()[0].isdigit() and input_text.split()[1].isdigit()):
            do_set_freq_limits('', input_text)
    elif event == 'Angle limits':
        from PictureBuilder import angle_from, angle_to
        input_text = sg.popup_get_text('Выбор пределов построения графика от[пиксель] до[пиксель] через пробел от 0 до 1199.\n' +
                                       'Сброс при вводе 0 0. Текущее значение ' + str(angle_from)+" "+str(angle_to))
        if (input_text and input_text.split()[0].isdigit() and input_text.split()[1].isdigit()):
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
        do_folder_preview(window, values["-CSV-"])
    elif event == 'Save .csv to ./Output':
        do_processing_plot('', mode='df')
    elif event == 'Save .pkl to ./Output':
        do_processing_plot('', mode='pkl')
    elif event == 'Find local max':
        do_processing_plot('', mode='find_max')
    elif event == 'Local max 3D (in this folder)':
        set_energy_limits(float(values["-ENERGYFROM-"]), float(values["-ENERGYTO-"]))
        do_processing_plot('', mode='3Den')
    elif event == 'Local max 3D (in all folders)':
        set_energy_limits(float(values["-ENERGYFROM-"]), float(values["-ENERGYTO-"]))
        do_processing_plot('', mode='3Ddistance')
    elif event == 'Energy contribution':
        do_processing_plot('', mode='energy_red')
    elif event == '_SLIDER_':
        window['_SHIFT_'].update(values['_SLIDER_']/10)
    elif event == '_SLIDERV_':
        window['_ANGLE_'].update(values['_SLIDERV_']/10.0)
    elif event == 'About...':
        window.disappear()
        sg.popup(' '*29+'~About this program~', 'Webpage: https://github.com/kitozawr/M150_freq_spectra',
                 "Help:        ./README.md", 'Author:     Zhidovtsev Nikita', '               PySimpleGUI rocks...',  grab_anywhere=True)
        window.reappear()

window.close()
