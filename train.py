import argparse

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import pytorch_lightning as pl

from model import GAN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='frog-dataset/data-224', type=str)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--workers', default=0, type=int)
    model_choices = ['dcgan']
    parser.add_argument('--gen', default='dcgan', type=str, choices=model_choices)
    parser.add_argument('--disc', type=str, choices=model_choices, required=False)
    parser.add_argument('--input_size', default=64, type=int)
    parser.add_argument('--latent', default=128, type=int)
    parser.add_argument('--features', default=64, type=int)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--accumulate', default=1, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    args = parser.parse_args()

    if not args.disc:
        args.disc = args.gen

    # Setup transforms
    normalize = T.Normalize(mean=[0.5], std=[0.5])
    real_transform = T.Compose([
        # T.Resize(args.input_size),
        # T.CenterCrop(args.input_size),
        T.RandomResizedCrop(args.input_size, scale=(0.92, 1.0), ratio=(0.95, 1.05), interpolation=T.InterpolationMode.NEAREST),
        # T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.05),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalize,
    ])
    real_ds = ImageFolder(root=args.root, transform=real_transform)
    real_loader = DataLoader(dataset=real_ds, batch_size=args.batch, num_workers=args.workers,
                    persistent_workers=(True if args.workers > 0 else False), pin_memory=True)

    import numpy as np
    from PIL import Image
    nrows = 9
    ncols = 16
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            idx = i*ncols + j + 2032
            x, y = real_ds[idx]
            x = (x+1)/2.0
            img = x.numpy().transpose(1, 2, 0)
            row.append(img)
        rows.append(np.hstack(row))
    grid_img = np.vstack(rows)*255
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.save("images/transform_examples.png")
    # im.show()
    # exit()

    # Setup models

    model = GAN(**vars(args))

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        gpus=args.ngpu,
        precision=args.precision,
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
    )

    trainer.fit(
        model=model,
        train_dataloader=real_loader,
    )

if __name__ == "__main__":
    main()


    