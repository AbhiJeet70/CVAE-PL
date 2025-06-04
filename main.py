import os

# Suppress TF INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.env import setup_environment
from src.downloader import download_data_sources
from src.data_pipeline import get_datasets, display_sample_images
from src.train import train_pvae, train_vae123
from src.visualization import display_history, display_rec_images, display_fake_images


def main():
    # 1) Kaggle-style environment setup
    setup_environment()

    # 2) Download all data sources (CelebA + VGG19 weights, etc.)
    download_data_sources()

    # 3) Build datasets and show a few sample training images
    train_ds, val_ds, test_ds = get_datasets()
    display_sample_images(train_ds)

    # 4) Hyperparameters
    alpha = 1.8
    beta = 10
    lr = 0.0015
    epochs = 30

    # 5) Train Plain VAE
    print("\n=== Training Plain VAE (pixel‐based) ===")
    pvae, history_pvae = train_pvae(alpha=alpha, beta=beta, lr=lr, epochs=epochs)
    display_history(history_pvae)

    # 6) Train VAE with Perceptual Loss
    print("\n=== Training VAE with Perceptual Loss (VGG19 features) ===")
    vae123, history_123 = train_vae123(alpha=alpha, beta=beta, lr=lr, epochs=epochs)
    display_history(history_123)

    # 7) Reconstruct on test set for both models
    print("\n=== Reconstructing test set (Plain VAE) ===")
    rec_dict_p = pvae.predict(test_ds)  # returns a dict with keys including 'reconstruction'
    test_rec_p = rec_dict_p['reconstruction']

    print("=== Reconstructing test set (VAE_123) ===")
    rec_dict_123 = vae123.predict(test_ds)
    test_rec_123 = rec_dict_123['reconstruction']

    img_path = '/kaggle/input/celeba-small-images-dataset/'

    # 8) Display original vs reconstructed for 5 example files
    print("\n--- Plain VAE Reconstructions ---")
    display_rec_images(test_rec_p, img_path)

    print("\n--- VAE_123 Reconstructions ---")
    display_rec_images(test_rec_123, img_path)

    # 9) Sample new “fake” faces from each decoder
    print("\n--- Sampling new faces: Plain VAE Decoder ---")
    display_fake_images(pvae.decoder)

    print("\n--- Sampling new faces: VAE_123 Decoder ---")
    display_fake_images(vae123.decoder)


if __name__ == "__main__":
    main()
