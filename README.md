# Celebrity Image Reconstruction using Deep Convolutional Variational AutoEncoder with Perceptual Loss (CVAE-PL)

In this study, two variants of Variational AutoEncoders (VAE) are investigated for their effectiveness in producing and reconstructing images using two different kinds of loss functions: feature perceptual loss (CVAE_PL) and pixel-by-pixel loss (PVAE) . The Jupyter Notebook with the name [Perceptual_Loss_in_VAE.ipynb](Perceptual_Loss_in_VAE.ipynb) contains the documentation and execution of the Python implementation.

# VAE Architecture with Perceptual Loss
![Autoencoder Network Architecture](images/VAE-with-perceptual-loss-architecture-overview.png)
# Encoder Decoder Architecture
![Autoencoder Network Architecture](images/VAE.png)
# Loss with PVAE
![Autoencoder Network Architecture](images/Loss_VAE.png)
# Loss with CVAE-PL
![Autoencoder Network Architecture](images/Loss_VAE123.png)
# Reconstruction with PVAE
![Autoencoder Network Architecture](images/reconstuction_with_plain_VAE.png)
# Reconstruction with CVAE_PL
![Autoencoder Network Architecture](images/reconstuction_with_VAE_123.png)
# New Face Generation with PVAE
![Autoencoder Network Architecture](images/NewFace_PVAE.png)
# New Face Generation with CVAE_PL
![Autoencoder Network Architecture](images/NewFace_VAE123.png)

### Dataset
For the investigations, the publicly accessible CelebFaces Attributes (CelebA) dataset is used.  For easy access, the dataset has been incorporated into the codebase. The CelebA dataset is described in further detail [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### How to Use
1. Make a local copy of the repository.
2. Launch and operate the Jupyter Notebook or Google Colab notebook [Perceptual_Loss_in_VAE.ipynb](Perceptual_Loss_in_VAE.ipynb) in your preferred environment.
3. Use the notebook's instructions to experiment with the VAE models and evaluate how well they work.

### Give Google Colab a try!
By clicking the link at the top of the notebook, you may quickly examine and run the source code in a Google Colab environment. You can also check [here](http://colab.research.google.com/github/AbhiJeet70/PerceptualLossVAE/blob/main/Perceptual_Loss_in_VAE.ipynb)



 
