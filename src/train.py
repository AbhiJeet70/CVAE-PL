import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

from .data_pipeline import get_datasets
from .models import create_preprocesser, create_encoder, create_decoder, PVAE, VAE_123

# You can adjust these default hyperparameters as needed:
DEFAULT_ALPHA = 1.8
DEFAULT_BETA = 10
DEFAULT_LR = 0.0015
DEFAULT_EPOCHS = 30


def train_pvae(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS):
    """
    Builds & trains the plain VAE (pixel‐based).
    Returns (pvae_model, history_object).
    """
    # 1) Get datasets
    train_ds, val_ds, _ = get_datasets()

    # 2) Build models
    preprocesser = create_preprocesser()
    encoder = create_encoder()
    decoder = create_decoder()
    pvae = PVAE(preprocesser, encoder, decoder, alpha, beta)

    # 3) Compile
    optimizer = Adam(lr)
    pvae.compile(optimizer=optimizer)

    # 4) Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss_wo_weight',
            factor=0.2,
            patience=2,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath='./pvae/',
            monitor="val_loss_wo_weight",
            verbose=1,
            save_best_only=True
        )
    ]

    # 5) Train
    history = pvae.fit(
        x=train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    return pvae, history


def train_vae123(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS):
    """
    Builds & trains the VAE with perceptual loss (using VGG19 features).
    Returns (vae123_model, history_object).
    """
    # 1) Get datasets
    train_ds, val_ds, _ = get_datasets()

    # 2) Build VGG19 feature extractor
    vgg19_base = VGG19(
        include_top=False,
        weights='/kaggle/input/vgg19-weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        input_shape=(64, 64, 3)
    )
    # We only want features from block1_conv1, block2_conv1, block3_conv1
    vgg19_123 = Model(
        inputs=vgg19_base.input,
        outputs=[
            vgg19_base.get_layer('block1_conv1').output,
            vgg19_base.get_layer('block2_conv1').output,
            vgg19_base.get_layer('block3_conv1').output
        ]
    )
    vgg19_123.trainable = False  # freeze VGG19

    # 3) Build VAE‐123
    preprocesser = create_preprocesser()
    encoder = create_encoder()
    decoder = create_decoder()
    vae123 = VAE_123(preprocesser, encoder, decoder, alpha, beta, vgg19_123)

    # 4) Compile (note: run_eagerly=True so that custom train_step works as intended)
    optimizer = Adam(lr)
    vae123.compile(optimizer=optimizer, run_eagerly=True)

    # 5) Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss_wo_weight',
            factor=0.2,
            patience=2,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath='./vae_123/',
            monitor="val_loss_wo_weight",
            verbose=1,
            save_best_only=True
        )
    ]

    # 6) Train
    history = vae123.fit(
        x=train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    return vae123, history
