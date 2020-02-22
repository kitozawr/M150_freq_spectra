import cmd
import sys
import os
import argparse
import numpy as np


def createParser ():
    parser = argparse.ArgumentParser(
        description = '''Обработка снимков камеры спектрометра. Возможности:
добавляет оси частоты и угла; применяет фильтры; может изменять график:
поворачивать, увеличивать, применять gain, изменять оси и название;
сохраняет в png; может открыть новый файл''',
        epilog = ''' (С) 2019 Nikita Zhidovtsev. Released under GNU GPL'''
        )
    parser.add_argument ('name', type=argparse.FileType(), nargs='?')
    parser.add_argument ('-f', '--frequency', type=int, help = 'Центральная длина волны спектрометра, нм')
    parser.add_argument ('-g', '--grating', type=int, help= 'Число штрихов решетки')
    parser.add_argument ('-s', '--scaletype', choices=['lin','log'], help='Выбор типа шкалы на графике. Варианты: {lin,log}', metavar= 'SCALE')
    parser.add_argument ('--rotate', action='store_const', const=True, help='Предварительный поворот на 180 градусов')
    parser.add_argument ('--title', help = 'Выбор названия на графике. По умолчанию используется имя файла')
    return parser

class CliInterface(cmd.Cmd):

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = "> "
        self.intro  = "Построение <<Тепловой карты>> для снимков камеры спектрометра. Для справки наберите 'help'"
        self.doc_header ="Доступные команды (для детальной справки по конкретной команде введите 'help _команда_'. Команды без справки являются служебными и не используются)"
    #Вызов функций будет осуществляться без 'do_'
    from PictureBuilder import do_set_freq_limits, do_processing_plot, do_processing_all_files_in_a_folder, do_set_rotate, do_list_clear_filters, do_save_filters, do_ask_add_filter,do_print_filters, do_list_add_filter as do_list_push_filter,do_list_rem_filter as do_list_pop_filter, do_set_angle_step,do_set_freq_step,do_print_parameters, do_set_grate,do_set_title,do_set_scale, do_set_freq, do_save_parameters, do_set_parameters, do_rotate as do_rotate_image, do_ask_save_file, do_ask_open_file, do_image_to_array, do_data_to_array,  do_plot, do_print_array, do_exit

if __name__ == "__main__":
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    cli= CliInterface()

    #Обработка имени файла
    if namespace.name:
        filename = namespace.name.name
        basepathname =os.path.basename(os.path.dirname(filename))
        cli.do_set_parameters(dirname= basepathname, pathname=os.path.dirname(filename), frequency=namespace.frequency, grating=namespace.grating,rotate=namespace.rotate, scaletype=namespace.scaletype, title=namespace.title)
        filename_extension =os.path.splitext(filename)[-1]
        if filename:
            if  filename_extension == ".png":
                cli.do_image_to_array(name_of_file=filename)
            elif filename_extension == ".dat":
                cli.do_data_to_array(name_of_file=filename)
            else:
                print("Недопустимое расширение файла")
    #Запуск бесконечного цикла
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print ("\nПринудительное выключение...")
