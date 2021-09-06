"""
Здесь хранятся функции ошибки для разных частей сети
"""


import torch


def generator_loss(x, y, generator, discriminator):
    generated = generator(x)
    data = torch.cat((x, generated), 1)
    dis_data = discriminator(data)
    loss = torch.nn.BCELoss(reduction="mean")(dis_data, torch.ones_like(dis_data))
    loss += torch.mean(torch.abs(generated - y)) * 100
    return loss


def discriminator_loss(x, y, generator, discriminator):
    generated = generator(x)
    dis_y = discriminator(torch.cat((x, y), 1))
    dis_generated = discriminator(torch.cat((x, generated), 1))
    loss_y = torch.nn.BCELoss(reduction='mean')(dis_y, torch.ones_like(dis_y))
    loss_generated = torch.nn.BCELoss(reduction='mean')(dis_generated, torch.zeros_like(dis_generated))
    return loss_y + loss_generated
