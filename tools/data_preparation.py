"""
Подготовка данных к обучению
"""

import torch

"""
Разделение данных на X и y 

enter: (len, 3, 256, 512)
out:   (len, 2, 3, 256, 256)
turned_add - удвоить датасет путем добавления в конец отзеркаленные картинки
"""


def split(x, turned_add=True, info=False):
    x_data = torch.FloatTensor()
    if turned_add:
         x_data = torch.FloatTensor(2*len(x), 2, 3, 256, 256)
    else:
        x_data = torch.FloatTensor(len(x), 2, 3, 256, 256)
    if info:
        l = int(len(x_data) / 20)
        total = -1
        print("%: ", end="")
    for i in range(len(x)):
        if info and i // l > total:
            print(i // l * 5, end=" ")
            total = i // l
        x_temp = x[i][0].unfold(2, 256, 256).permute(2, 0, 1, 3)
        x_data[i][0] = x_temp[0]
        x_data[i][1] = x_temp[1]
        if turned_add:
            x_data[len(x) + i][0] = torch.flip(x_temp[0], [2])
            x_data[len(x) + i][1] = torch.flip(x_temp[1], [2])

    if info:
        print("\nfinished")
    return x_data


"""
Перемещение данных на устройство
"""


def move_to(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to(x, device) for x in data]
    return data.to(device, non_blocking=True)
