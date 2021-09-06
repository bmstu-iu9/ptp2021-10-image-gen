from tkinter import *
import torch
import pickle
import PIL.ImageTk
import PIL.Image as Img
from PIL import ImageDraw
import torchvision.transforms as tt

from tools import data_preparation as dp
from torchvision.datasets import ImageFolder

def load(name="model.pkl"):
    with open(name, "rb") as f:
        return pickle.load(f)


class Paint(Frame):
    def __init__(self, parent, gen):
        Frame.__init__(self, parent)
        self.parent = parent
        self.setUI()
        self.brush_size = 1
        self.brush_color = "black"
        self.canv.bind("<B1-Motion>", self.draw)
        self.image = Img.new("RGB", (256, 256), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.gen = gen


    def setUI(self):
        self.parent.title("Pythonicway PyPaint")
        self.pack(fill=BOTH, expand=1)

        self.canv = Canvas(self, bg="white", width=256, height=256)
        self.canv.grid(row=1, column=0, columnspan=4, padx=5, pady=3)

        self.canv2 = Canvas(self, bg="white", width=256, height=256)
        self.canv2.grid(row=1, column=4, padx=1, pady=1)

        brush_btn = Button(self, text="Brush", width=10, command=lambda: self.set_color("black", 1))
        brush_btn.grid(row=0, column=0)

        erase_btn = Button(self, text="Eraser", width=10, command=lambda: self.set_color("white", 7))
        erase_btn.grid(row=0, column=1)

        clean_btn = Button(self, text="Clean", width=10, command=lambda: self.clean())
        clean_btn.grid(row=0, column=2)

        execute_btn = Button(self, text="Run", width=10, command=lambda: self.get())
        execute_btn.grid(row=0, column=3)

    def draw(self, event):
        x1, x2, y1, y2 = event.x, event.x + self.brush_size, event.y, event.y + self.brush_size
        self.canv.create_oval(x1, y1, x2, y2, fill=self.brush_color, outline=self.brush_color)  # для отображения
        self.draw.ellipse([(x1, y1), (x2, y2)], fill=self.brush_color, outline=self.brush_color)  # для записи

    def set_color(self, color, bsize):
        self.brush_color = color
        self.brush_size = bsize

    def clean(self):
        self.canv.delete("all")
        self.image = Img.new("RGB", (256, 256), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def get(self):
        a = tt.ToTensor()(self.image)
        img = torch.FloatTensor(1, 3, 256, 256)
        img[0] = a


        with torch.no_grad():
            img = tt.ToPILImage()(self.gen(img)[0])


        to_show = PIL.ImageTk.PhotoImage(img)  # приводим PILImage к TkinterImage
        # img.show()
        self.canv2.create_image(128, 128, image=to_show)
        self.parent.mainloop()

