
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from model import GAN


# path = "lightning_logs/version_16/checkpoints/epoch=199-step=24399.ckpt"
# path = "lightning_logs/version_17/checkpoints/epoch=344-step=10670.ckpt"
# path = "lightning_logs/version_18/checkpoints/epoch=799-step=24799.ckpt"
# path = "lightning_logs/version_19/checkpoints/epoch=399-step=6399.ckpt"
# path = "lightning_logs/version_21/checkpoints/epoch=399-step=6399.ckpt"
# path = "lightning_logs/version_28/checkpoints/epoch=1199-step=73199.ckpt"
# path = "lightning_logs/version_29/checkpoints/epoch=1799-step=55799.ckpt"

path = "lightning_logs/version_37/checkpoints/epoch=799-step=24799.ckpt"
# path = "lightning_logs/version_38/checkpoints/epoch=999-step=30999.ckpt"
# path = "lightning_logs/version_39/checkpoints/epoch=1999-step=61999.ckpt"
# path = "lightning_logs/version_49/checkpoints/epoch=1999-step=61999.ckpt"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GAN.load_from_checkpoint(path, map_location=device)

samples = 144

gen_imgs = model.sample(samples)
print(gen_imgs.shape)
grid = torchvision.utils.make_grid(gen_imgs, nrow=16)
# grid = F.interpolate(grid.unsqueeze_(0), scale_factor=2, mode='bilinear', align_corners=False).squeeze()
print(grid.shape)
print(torch.min(grid), torch.max(grid))

grid = (grid+1)/2.0
print(torch.min(grid), torch.max(grid))

im = Image.fromarray((grid*255).numpy().transpose(1, 2, 0).astype('uint8'), 'RGB')
im.show()
im.save("images/frogs21.png")

# plt.imshow(grid.permute(1, 2, 0))
# plt.show()
