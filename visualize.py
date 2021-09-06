import pickle
import torch
from tkinter import *
import tools.display as display
from tools import data_preparation


def load(name="model.pkl"):
    with open(name, "rb") as f:
        return pickle.load(f)


gen = data_preparation.move_to(load("models/lgenerator.pkl"), torch.device("cpu"))
#gen.eval()

root = Tk()
root.geometry("610x300")
app = display.Paint(root, gen)
root.mainloop()


"""
# альтернативный вариант
import torch
import torchvision.transforms as tt
import PIL.Image as Img

a = Img.open("test.png")
a = tt.ToTensor()(a)[:3]

img = torch.FloatTensor(1, 3, 256, 256)
img[0] = a

with torch.no_grad():
    img = tt.ToPILImage()(self.gen(img)[0])

img.show()

"""




