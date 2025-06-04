import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def display_history(history):
    """
    Given a Keras History object → plot reconstruction_loss and kl_loss over epochs.
    """
    fig = plt.figure(figsize=(20, 5))

    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history['reconstruction_loss'])
    plt.plot(history.history['val_reconstruction_loss'])
    plt.title('reconstruction_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history['kl_loss'])
    plt.plot(history.history['val_kl_loss'])
    plt.title('kl_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()


def display_rec_images(test_ds_reconstructed, img_path):
    """
    Given a list/array of reconstructed images (numpy uint8),
    plot 5 originals (hardcoded filenames) vs. their reconstructions.

    NOTE: The filenames below match the original code’s examples. Adjust if needed.
    """
    images = ['182638.jpg', '182639.jpg', '182640.jpg', '182641.jpg', '182642.jpg']

    fig = plt.figure(figsize=(20, 5))

    for i in range(5):
        # original
        f = img_path + 'testing/' + images[i]
        img = Image.open(f)
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')

        # reconstructed (assumes test_ds_reconstructed is a Numpy array)
        ax = plt.subplot(2, 5, i + 6)
        plt.imshow(test_ds_reconstructed[i].astype("uint8"))
        plt.axis('off')

    plt.show()


def display_fake_images(decoder):
    """
    Sample 10 random z ~ N(0, 0.5²) and pass through `decoder` to generate fake faces.
    """
    z = np.random.normal(loc=0.0, scale=0.5, size=(10, 100))
    fakefaces = decoder(z)

    plt.figure(figsize=(20, 5))
    for i in range(len(fakefaces)):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(fakefaces[i].numpy().astype("uint8"))
        plt.axis('off')
    plt.show()
