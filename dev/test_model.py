
import torch
import torch.nn as nn
import torchvision

from model import Generator, Discriminator

input_size = 64
latent = 128
features = 64

gen = Generator(latent, features, input_size)

# print(gen); exit()

batch = 1
z = torch.randn((batch, latent, 1, 1))
fake = gen(z)
print("z :", z.shape)
print("fake :", fake.shape)
grid = torchvision.utils.make_grid(fake)
grid = (grid+1)/2.0

print(grid.shape)
print(torch.min(grid), torch.max(grid))

import matplotlib.pyplot as plt
plt.imshow(grid.detach().permute(1, 2, 0))
# plt.show()


# print(gen.gen[0][0].weight)

disc = Discriminator(features)
yhat = disc(fake)
print("yhat :", yhat.shape)




