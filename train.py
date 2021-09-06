import torch  # для работы с данными
from torchvision.datasets import ImageFolder  # для загрузки данных
from torch.utils.data import DataLoader  # для деления данных на батчи при обучении

from tools import tools  # составные архитектуры
import tools.data_preparation as dp  # для обработки данных
import tools.losses as loss  # для функций ошибки
import torchvision.transforms as tt  # для обработки данных
import pickle  # для сохранения / загрузки модели


from torchvision.datasets.utils import download_file_from_google_drive


def save(to_save, name="model.pkl"):
    with open(name, "wb") as f:
        pickle.dump(to_save, f)


def load(name="model.pkl"):
    with open(name, "rb") as f:
        return pickle.load(f)


data = ImageFolder('./dataset/val', transform=tt.Compose([
  tt.ToTensor()
]))


gen = tools.GeneratorLight()
dis = tools.Discriminator()

gen.train()

data = dp.split(data, turned_add=True, info=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# перемещаем модели и данные на устройство
gen = dp.move_to(gen, device)
dis = dp.move_to(dis, device)
data = dp.move_to(data, device)


"""
Обучение
part - часть данных на которых происходит обучение (от 0 до 1: соответственно 0% - 100%)
glpdl - итераций обучения генератора на итерацию обучения дискриминатора
backup - нужно ли делать промежуточные бэкапы при обучении
backup_rate - частота в эпохах, когда делается резервная копия генератора
lr_gen и lr_dis - learning rate используемая при оптимизации
"""
part = 0.1
epochs = 10
batch_size = 4

lr_gen = 2e-4
lr_dis = 2e-4

glpdl = 1

backup = True
backup_rate = 10


part_learn = int(len(data) * part)

if batch_size > part_learn:
    raise()  # здесь кидает исключение что батч больше данных


x_loader = DataLoader(data[:part_learn], batch_size=batch_size, drop_last=True, shuffle=True)

X = next(iter(x_loader))[:, 0]
y = next(iter(x_loader))[:, 1]
enter = torch.cat((X, y), 1)
res = dis(enter)[0].permute(1, 2, 0).detach()
res = res.view(30, 30)

gen_optim = torch.optim.Adam(gen.parameters(), lr=lr_gen)
dis_optim = torch.optim.Adam(dis.parameters(), lr=lr_dis)


gen_losses = []
dis_losses = []
pics = 0

for epoch in range(epochs):
    print(">> ", epoch, " | ", sep="", end="")
    print("from ", part_learn, "p : ", sep="", end="")
    for data in x_loader:
        X = data[:, 0]
        y = data[:, 1]

        dis_optim.zero_grad()
        dis_loss = loss.discriminator_loss(X, y, gen, dis)
        dis_loss.backward()
        dis_optim.step()

        gen_loss = 0
        for i in range(glpdl):
            gen_optim.zero_grad()
            gen_loss = loss.generator_loss(X, y, gen, dis)
            gen_loss.backward()
            gen_optim.step()

        gen_losses.append(float(gen_loss))
        dis_losses.append(float(dis_loss))

        pics += batch_size
        print(pics, end="p ")

    if backup and (epoch + 1) % backup_rate == 0:
        gen_cpu = dp.move_to(gen, torch.device("cpu"))
        save(gen_cpu, "models/backup/backup_" + str(epoch // backup_rate) + ".pkl")
        dp.move_to(gen, device)

    print()
    print("finished with Gen Loss: ", float(gen_loss), " ,Dis Loss: ", float(dis_loss))

    pics = 0

save(dp.move_to(gen, torch.device("cpu")), "models/lgenerator1.pkl")


