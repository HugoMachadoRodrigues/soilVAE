import keras
import tensorflow as tf

class VAEReg(keras.Model):
    def __init__(self, input_dim, hidden_enc=(512,256), hidden_dec=(256,512),
                 latent_dim=32, dropout=0.1, beta_kl=1.0, alpha_y=1.0, name='vae_reg'):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.beta_kl = float(beta_kl)
        self.alpha_y = float(alpha_y)

        # ----- Encoder -----
        self.enc_layers = []
        for u in hidden_enc:
            self.enc_layers.append(keras.layers.Dense(int(u), activation='relu'))
            if dropout and dropout > 0:
                self.enc_layers.append(keras.layers.Dropout(float(dropout)))

        self.z_mean = keras.layers.Dense(self.latent_dim, name='z_mean')
        self.z_logv = keras.layers.Dense(self.latent_dim, name='z_logv')

        # ----- Decoder -----
        self.dec_layers = []
        for u in hidden_dec:
            self.dec_layers.append(keras.layers.Dense(int(u), activation='relu'))
            if dropout and dropout > 0:
                self.dec_layers.append(keras.layers.Dropout(float(dropout)))

        self.x_hat = keras.layers.Dense(self.input_dim, activation='linear', name='x_hat')

        # ----- y head -----
        self.y_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ], name='y_head')

        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker    = keras.metrics.Mean(name='kl_loss')
        self.y_loss_tracker     = keras.metrics.Mean(name='y_loss')

    def encode(self, x, training=False):
        h = x
        for layer in self.enc_layers:
            h = layer(h, training=training)
        z_mean = self.z_mean(h)
        z_logv = self.z_logv(h)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_logv) * eps
        return z, z_mean, z_logv

    def decode(self, z, training=False):
        h = z
        for layer in self.dec_layers:
            h = layer(h, training=training)
        return self.x_hat(h)

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker,
                self.y_loss_tracker]

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        with tf.GradientTape() as tape:
            z, z_mean, z_logv = self.encode(x, training=True)
            x_hat = self.decode(z, training=True)
            y_hat = self.y_head(z, training=True)

            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + z_logv - tf.square(z_mean) - tf.exp(z_logv), axis=1))
            y_loss = tf.reduce_mean(tf.square(y - y_hat))

            total = recon_loss + self.beta_kl * kl + self.alpha_y * y_loss

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        self.y_loss_tracker.update_state(y_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        z, z_mean, z_logv = self.encode(x, training=False)
        x_hat = self.decode(z, training=False)
        y_hat = self.y_head(z, training=False)

        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + z_logv - tf.square(z_mean) - tf.exp(z_logv), axis=1))
        y_loss = tf.reduce_mean(tf.square(y - y_hat))
        total = recon_loss + self.beta_kl * kl + self.alpha_y * y_loss

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        self.y_loss_tracker.update_state(y_loss)

        return {m.name: m.result() for m in self.metrics}
