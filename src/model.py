"""
src/model.py
============
ConvLSTM Autoencoder architecture for video anomaly detection.
Import this module to rebuild the model from scratch for re-training.
"""

from __future__ import annotations


def build_model(seq_len: int = 10, frame_h: int = 256, frame_w: int = 256):
    """
    Build and return the ConvLSTM Autoencoder.

    Architecture
    ------------
    Encoder  : Two TimeDistributed Conv2D layers progressively downsample
               spatial resolution while extracting local features.
    Temporal : Three stacked ConvLSTM2D layers capture motion dynamics
               across the 10-frame input window.
    Decoder  : Mirrored ConvTranspose2D layers upsample back to original
               resolution, ending with a sigmoid Conv2D.

    Parameters
    ----------
    seq_len  : number of frames in each input clip  (default 10)
    frame_h  : frame height in pixels               (default 256)
    frame_w  : frame width  in pixels               (default 256)

    Returns
    -------
    tf.keras.Model  (un-compiled)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    inp = Input(shape=(seq_len, frame_h, frame_w, 1), name="input_sequence")

    # ── Encoder ────────────────────────────────────────────────────────────
    x = layers.TimeDistributed(
        layers.Conv2D(128, (11, 11), strides=4, padding="same",
                      activation="relu"), name="enc_conv1")(inp)
    x = layers.LayerNormalization(name="enc_ln1")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(64, (5, 5), strides=2, padding="same",
                      activation="relu"), name="enc_conv2")(x)
    x = layers.LayerNormalization(name="enc_ln2")(x)

    # ── Temporal / ConvLSTM core ────────────────────────────────────────────
    x = layers.ConvLSTM2D(64, (3, 3), padding="same",
                          return_sequences=True, name="lstm1")(x)
    x = layers.LayerNormalization(name="lstm_ln1")(x)

    x = layers.ConvLSTM2D(32, (3, 3), padding="same",
                          return_sequences=True, name="lstm2")(x)
    x = layers.LayerNormalization(name="lstm_ln2")(x)

    x = layers.ConvLSTM2D(64, (3, 3), padding="same",
                          return_sequences=True, name="lstm3")(x)
    x = layers.LayerNormalization(name="lstm_ln3")(x)

    # ── Decoder ────────────────────────────────────────────────────────────
    x = layers.TimeDistributed(
        layers.Conv2DTranspose(64, (5, 5), strides=2, padding="same",
                               activation="relu"), name="dec_conv1")(x)
    x = layers.LayerNormalization(name="dec_ln1")(x)

    x = layers.TimeDistributed(
        layers.Conv2DTranspose(128, (11, 11), strides=4, padding="same",
                               activation="relu"), name="dec_conv2")(x)
    x = layers.LayerNormalization(name="dec_ln2")(x)

    out = layers.TimeDistributed(
        layers.Conv2D(1, (1, 1), activation="sigmoid"), name="output")(x)

    return Model(inp, out, name="ConvLSTM_Autoencoder")


def compile_model(model, learning_rate: float = 1e-4, decay: float = 1e-5):
    """Compile the model with Adam + MSE loss."""
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, decay=decay
        ),
        loss="mse",
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
