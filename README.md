# pix2fun
![logo](https://github.com/bmstu-iu9/ptp2021-10-image-gen/blob/main/logo.png)

###Генерация изображений по очертаниям

Начертите контур предмета и модель постарается построить картинку определенного типа по вашему рисунку
![example](https://github.com/bmstu-iu9/ptp2021-10-image-gen/blob/main/example.png)

### Начало работы

Для корректной работы понадобится интерпретатор python3 и установленный библеотеки torch и torchvision </br>
Подробнее об установке здесь: https://pythonworld.ru/osnovy/pip.html

В корне проекта понадобятся папки: "dataset" куда вы можете загрузить свой датасет по формату </br> или скачать отсюда: https://www.kaggle.com/vikramtiwari/pix2pix-dataset

и папка "models" с папкой "backup" внутри, в папку "models" хранятся модели, куда вы можете загрузить модель </br> например отсюда: https://drive.google.com/file/d/1UIuTo2vIVkXYvyMMPmsltg8rpnnrG3Ur/view?usp=sharing

### Функционал

Исполните visualize.py для визуализации генерации для модели из models, модель и параметры настройте в программе

Исполните train.py для создания модели по данным, данные, параметры обучения и др настройте в программе

### Список участников

* Наумов Сергей - <a href=https://github.com/pear2jam> @pear2jam </a> 
