import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend


def create_preprocesser():
    """
    A simple model that takes input images → rescales them to [0,1].
    """
    inputs = keras.Input(shape=(64, 64, 3))
    preprocessed = layers.Rescaling(1.0 / 255)(inputs)
    return keras.Model(inputs, preprocessed, name='preprocesser')


def create_encoder():
    """
    Builds a 64×64×3 → latent encoder that outputs (z_mean, z_log_sigma, z).
    """
    inputs = keras.Input(shape=(64, 64, 3))

    # 32×4×4 conv, stride 2 + BN + LeakyReLU
    x = layers.Conv2D(32, (4, 4), strides=2, padding='same', activation=None)(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.LeakyReLU(alpha=0.3)(x)

    # 64×4×4 conv, stride 2 + BN + LeakyReLU
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 128×4×4 conv, stride 2 + BN + LeakyReLU
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 256×4×4 conv, stride 2 + BN + LeakyReLU
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Flatten and produce latent parameters
    x = layers.Flatten()(x)
    z_mean = layers.Dense(units=100, activation=None)(x)
    z_log_sigma = layers.Dense(units=100, activation=None)(x)

    # Sampling layer: z = z_mean + exp(z_log_sigma/2) * ε
    def sampling(args):
        z_mean_, z_log_sigma_ = args
        epsilon = backend.random_normal(shape=(1, 100), mean=0.0, stddev=0.1)
        return z_mean_ + backend.exp(z_log_sigma_ / 2) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    return keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')


def create_decoder():
    """
    Takes a latent vector (100,) → upsamples back to 64×64×3.
    """
    inputs = keras.Input(shape=(100,))

    x = layers.Dense(units=4 * 4 * 256, activation='relu')(inputs)
    x = layers.Reshape((4, 4, 256))(x)

    # Upsample + conv128×3×3 + BN + LeakyReLU
    x = layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Upsample + conv64×3×3 + BN + LeakyReLU
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Upsample + conv32×3×3 + BN + LeakyReLU
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Final upsample + conv3×3 → RGB output
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), padding='same')(x)

    return keras.Model(inputs, decoded, name='decoder')


class PVAE(keras.Model):
    """
    Plain VAE (pixel‐based). Loss = α * MSE(data, recon) + β * KL.
    """

    def __init__(self, preprocesser, encoder, decoder, alpha, beta, **kwargs):
        super(PVAE, self).__init__(**kwargs)
        self.preprocesser = preprocesser
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta

        # Metrics to track
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_wo_weight_tracker = keras.metrics.Mean(name="loss_wo_weight")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.loss_wo_weight_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def forward(self, data):
        """
        data: raw images (uint8), shape (batch, 64,64,3).
        returns (z_mean, z_log_sigma, reconstruction).
        """
        x = self.preprocesser(data)
        z_mean, z_log_sigma, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_sigma, reconstruction

    def compute_loss(self, data, z_mean, z_log_sigma, reconstruction):
        """
        MSE pixel loss + KL divergence.
        """
        # Pixel reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(tf.cast(data, tf.float32), reconstruction)
        )
        # KL divergence
        kl_loss = -0.5 * (1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        loss_wo_weight = reconstruction_loss + kl_loss
        loss = self.alpha * reconstruction_loss + self.beta * kl_loss
        return reconstruction_loss, kl_loss, loss, loss_wo_weight

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_sigma, reconstruction = self.forward(data)
            rec_loss, kl_loss, loss, loss_wo_weight = self.compute_loss(data, z_mean, z_log_sigma, reconstruction)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.loss_wo_weight_tracker.update_state(loss_wo_weight)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.loss_tracker.result(),
            "loss_wo_weight": self.loss_wo_weight_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, data):
        """
        Used for validation: returns metrics + reconstruction so we can inspect.
        """
        z_mean, z_log_sigma, reconstruction = self.forward(data)
        rec_loss, kl_loss, loss, loss_wo_weight = self.compute_loss(data, z_mean, z_log_sigma, reconstruction)

        self.loss_tracker.update_state(loss)
        self.loss_wo_weight_tracker.update_state(loss_wo_weight)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.loss_tracker.result(),
            "loss_wo_weight": self.loss_wo_weight_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "reconstruction": reconstruction,
        }


class VAE_123(keras.Model):
    """
    VAE with perceptual (VGG19‐based) loss. Loss = α * perceptual_loss + β * KL.
    """

    def __init__(self, preprocesser, encoder, decoder, alpha, beta, vgg19_123, **kwargs):
        super(VAE_123, self).__init__(**kwargs)
        self.preprocesser = preprocesser
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        self.vgg19_123 = vgg19_123

        # Metrics to track
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_wo_weight_tracker = keras.metrics.Mean(name="loss_wo_weight")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.loss_wo_weight_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=False):
        """
        If training=True → compute and return losses.
        Else → return just the reconstructed image.
        """
        x = self.preprocesser(inputs)
        z_mean, z_log_sigma, z = self.encoder(x)
        recon = self.decoder(z)

        if training:
            return self.compute_loss(inputs, z_mean, z_log_sigma, recon)
        else:
            return recon

    def compute_loss(self, data, z_mean, z_log_sigma, reconstruction):
        """
        Perceptual loss: pass both data & reconstruction through VGG19 up to certain layers,
        compute MSE in that feature space, average over 2 layers, then add KL.
        """
        # Preprocess both for VGG19
        data_pp = tf.keras.applications.vgg19.preprocess_input(tf.cast(data, tf.float32))
        recon_pp = tf.keras.applications.vgg19.preprocess_input(reconstruction)

        # Get feature maps from chosen layers
        v1_data, v2_data, v3_data = self.vgg19_123(data_pp)
        v1_recon, v2_recon, v3_recon = self.vgg19_123(recon_pp)

        # MSE in the VGG19 feature space (averaging first two chosen layers)
        rec_loss_1 = tf.reduce_mean(keras.losses.mean_squared_error(v1_data, v1_recon))
        rec_loss_2 = tf.reduce_mean(keras.losses.mean_squared_error(v2_data, v2_recon))
        reconstruction_loss = (rec_loss_1 + rec_loss_2) / 2

        # KL divergence
        kl_loss = -0.5 * (1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        loss_wo_weight = reconstruction_loss + kl_loss
        loss = self.alpha * reconstruction_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "loss_wo_weight": loss_wo_weight,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.call(data, training=True)
            loss = loss_dict["loss"]
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss_dict["loss"])
        self.loss_wo_weight_tracker.update_state(loss_dict["loss_wo_weight"])
        self.reconstruction_loss_tracker.update_state(loss_dict["reconstruction_loss"])
        self.kl_loss_tracker.update_state(loss_dict["kl_loss"])

        return loss_dict

    def test_step(self, data):
        # Similar logic for validation
        loss_dict = self.call(data, training=True)
        return loss_dict
