Dataset link: https://github.com/jonshamir/frog-dataset

Video Demo: [2 - Timelapse of DCGAN learning to make frogs](https://www.youtube.com/watch?v=pSHeBhLKc4w)

Extract the images generated during training from the tensorboard logs and make video: dev/tb_images.py

Findings:
- model.py L89 - reducing the discriminator learning rate helps. unclear why
- dcgan.py - discriminator with spectral_norm
- dcgan.py - activations that are not piecewise work better. I think it's because there is more gradient signal. ie. GELU and SiLU are better than ReLU
- train.py - RandomResizedCrop with limited scaling for data augmentation


frog-dataset notes
```
turtle images
raw/frog-7542
224/frog-5727
64/frog-5727

what is this
224/frog-7662
64/frog-7662

4 frogs
raw/frog-7716
```
