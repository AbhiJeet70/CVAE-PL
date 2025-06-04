import tensorflow as tf

# Constants
batch_size = 64
img_path = '/kaggle/input/celeba-small-images-dataset/'


def process_path(file_path):
    """
    Given a file path string → load + decode JPEG → return raw image tensor.
    (For an autoencoder, the “label” is the image itself; we return just `img`.)
    """
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img


def configure_for_performance(ds):
    """
    Cache, batch, and prefetch for performance.
    """
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_datasets():
    """
    Returns three tf.data.Datasets: (train_ds, val_ds, test_ds),
    each yielding raw image tensors of shape (h, w, 3).
    """
    train_ds = tf.data.Dataset.list_files(img_path + 'training/*.jpg', shuffle=False)
    val_ds = tf.data.Dataset.list_files(img_path + 'validation/*.jpg', shuffle=False)
    test_ds = tf.data.Dataset.list_files(img_path + 'testing/*.jpg', shuffle=False)

    # Map each path to its raw image tensor
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    return train_ds, val_ds, test_ds


def display_sample_images(train_ds):
    """
    Shows 10 sample images from one batch of train_ds.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for image_batch in train_ds.take(1):
        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()
