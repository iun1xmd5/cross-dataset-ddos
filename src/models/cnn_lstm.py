#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN-LSTM hybrid: a Conv1D feature extractor followed by a single LSTM layer."""

from typing import Tuple

from config import require_keras


def build_cnn_lstm(input_shape: Tuple[int, int]):
    """
    Conv1D(64, k=3) -> MaxPool(2) -> LSTM(64) -> batch-norm/dropout -> Dense(64) -> sigmoid.
    The lightest of the three neural models.

    ``input_shape`` is (window, n_features).
    """
    tf, layers, _, _, _ = require_keras()
    inp = tf.keras.Input(shape=input_shape, name="sequence_input")
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)
    return tf.keras.Model(inp, out, name="CNN-LSTM")
