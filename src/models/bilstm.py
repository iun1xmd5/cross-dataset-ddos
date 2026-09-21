#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stacked bidirectional LSTM classifier (256-128-64) for sequences of network flows."""

from typing import Tuple

from config import require_keras


def build_bilstm(input_shape: Tuple[int, int]):
    """
    Same layout as the LSTM model, but every recurrent layer is bidirectional, so each layer
    reads the window forwards and backwards (twice the recurrent parameters).

    ``input_shape`` is (window, n_features).
    """
    tf, layers, _, _, regularizers = require_keras()
    reg = regularizers.l2(1e-4)
    inp = tf.keras.Input(shape=input_shape, name="sequence_input")
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, kernel_regularizer=reg))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, kernel_regularizer=reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)
    return tf.keras.Model(inp, out, name="BiLSTM")
