

import torch


from model import Discriminator


model = Discriminator(16)

data = torch.randn((8, 3, 128, 64))
print(data.shape)

yhat = model(data)
print(yhat.shape, yhat.dtype)

valid = torch.ones_like(yhat)
print(valid.shape, valid.dtype)








