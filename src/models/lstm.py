#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stacked LSTM classifier (256-128-64) for sequences of network flows."""

from typing import Tuple

from config import require_keras


def build_lstm(input_shape: Tuple[int, int]):
    """
    Three stacked LSTM layers (256 -> 128 -> 64) with batch-norm and dropout, followed by a
    128-unit dense layer and a sigmoid output (P(ATTACK)).

    ``input_shape`` is (window, n_features). The output layer is kept in float32 so the
    model also works under mixed-precision training.
    """
    tf, layers, _, _, regularizers = require_keras()
    reg = regularizers.l2(1e-4)
    inp = tf.keras.Input(shape=input_shape, name="sequence_input")
    x = layers.LSTM(256, return_sequences=True, kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)
    return tf.keras.Model(inp, out, name="LSTM")
