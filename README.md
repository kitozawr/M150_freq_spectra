Данная справка актуальна только для консольной версии. Графическая версия реализует legacy данный функционал, заменяя spectrograph.py


# Построение ЧУС
Модуль реализует открытие и разметку файлов камеры спектрометра.

## Работа с файлами конфигурации
_При изменении пути к модулю необходимо изменить переменные "address..."_

Модуль создает файл (указан в "address_") для запоминания последней открытой директории через "ask_" и папки с фильтрами. Файлы фильтра это .txt с двумя столбцами <длина_волны, нм> + <коэфициент_пропускания>, где используется точка для разделения float (123.45). В папках с файлами кадров камеры спектрометра сохраняются файлы с именем "spectrograph ... .pkl" в которых содержится информация о выбранной решетке и прочих деталях режима, использованного в этих измерениях.

Введенные параметры можно сохранить в папке файла при помощи команды "save_parameters". Далее они будут загружаться при каждом обращении к файлам этой папки. Для учета нового фильтра можно ввести его имя как аргумент "list_add_filter" или через графическое окно "ask_add_filter". При помощи "print_filters" можно узнать о фильтрах, которые уже учтены. Для удаления используйте "list_rem_filter", который принимает номер из вывода "print_filters". Спектр поглощения камеры и решеток учитывать не надо, он добавляется в расчет автоматически на этапе построения графика.

## Работа с данными
О способах открытия нового файла и изменения параметров смотри справку к spectrograph.py, который реализует наиболее удобный из них. Под _параметрами_ здесь подразумеваются указанные в названии папки настройки спектрографа. 

Для построения спектра вызовите "plot". Обновление данных происходит при открытии нового графика при помощи функции "plot". Перед этим предыдущий график необходимо закрыть. Сохранение спектра доступно как через графический, так и консольный интерфейс.

Возможны задержки открытия файла по CTRL-C при построенном графике (идет закрытие окон). Для ускорения процесса (долгий вызов закрытия gui-окон) нажмите ENTER или используйте команду exit()

Фон вычисляется как среднее в рамке на границе изображжения. Отрицательные значения зануляются.

## spectrograph.py
Создает CLI интерфейс к модулю __PictureBuilder__, позволяя запускать некоторые его функции. Доступен autocomplete и история команд. Выход по CTRL+C или команде exit. Введите "spectrograph --help" для получения данных о аргуметах командной строки. После запуска введите "help" для получения справки о доступных командах и "help команда" для справки о каждой из них

Для открытия изображения и ввода его свойств есть три доступных пути:
* После запуска при помощи команды "ask_open_file" (команды "ask_" указывают на запуск графических окон для выбора). Нельзя указать дополнительные аргументы. Найденные в папке настройки будут загружены в приоритете. При их отсутствии выставлены по умолчанию. Выставить новые параметры можно будет при помощи группы функций "set_". Рекомендуемый способ, реализуемый данным скриптом.
* Через командную строку: "spectrograph -f 800 -g 300". Аргументы командной строки (частота, решетка и тп) будут учтены в приоритетном порядке. Если в папке файла будут найдены сохраненные настройки они будут учтены в менее приоритетном порядке. Путь к изображению будет использован для поиска фильтров. Данный способ поддерживается, но более не разрабатывается.
* Использование команд image_to_array для PNG и data_to_array для .dat. Только если известно точное имя файла в папке последнего открытого изображения. Параметры и фильтры не изменяются. Автодополнение отсутствует. Настройки не изменяются. Выставить новые параметры можно будет только при помощи группы функций "set_". Не рекомендуется

О более подробном описании функций модуля смотри справку к __PictureBuilder__ и команду "help <функция>"
