# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A TensorFlow model for identifying exoplanets in astrophysical light curves.

AstroModel is a concrete base class for models that identify exoplanets in
astrophysical light curves. This class implements a simple linear model that can
be extended by subclasses.

The general framework for AstroModel and its subclasses is as follows:

  * Model inputs:
     - Zero or more time_series_features (e.g. astrophysical light curves)
     - Zero or more aux_features (e.g. orbital period, transit duration)

  * Labels:
     - An integer feature with 2 or more values (eg. 0 = Not Planet, 1 = Planet)

  * Model outputs:
     - The predicted probabilities for each label

  * Architecture:

                         predictions
                              ^
                              |
                           logits
                              ^
                              |
                   (pre_logits_hidden_layers)
                              ^
                              |
                       pre_logits_concat
                              ^
                              |
                        (concatenate)
                ^                           ^
                |                           |
     (time_series_hidden_layers)    (aux_hidden_layers)
                ^                           ^
                |                           |
       time_series_features           aux_features


Subclasses will typically override the build_time_series_hidden_layers()
and/or build_aux_hidden_layers() functions. For example, a subclass could
override build_time_series_hidden_layers() to apply convolutional layers to the
time series features. In this class, those functions are simple concatenations
of the input features.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import tensorflow as tf


class AstroModel(object):
  """A TensorFlow model for classifying astrophysical light curves."""

  def __init__(self, features, labels, hparams, mode):
    """Basic setup.

    The actual TensorFlow graph is constructed in build().

    Args:
      features: A dictionary containing "time_series_features" and
        "aux_features", each of which is a dictionary of named input Tensors.
        All features have dtype float32 and shape [batch_size, length].
      labels: An int64 Tensor with shape [batch_size]. May be None if mode is
        tf.estimator.ModeKeys.PREDICT.
      hparams: A ConfigDict of hyperparameters for building the model.
      mode: A tf.estimator.ModeKeys to specify whether the graph should be built
        for training, evaluation or prediction.

    Raises:
      ValueError: If mode is invalid.
    """
    valid_modes = [
        tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
        tf.estimator.ModeKeys.PREDICT
    ]
    if mode not in valid_modes:
      raise ValueError("Expected mode in {}. Got: {}".format(valid_modes, mode))

    self.hparams = hparams
    self.mode = mode

    # A dictionary of input Tensors. Values have dtype float32 and shape
    # [batch_size, length].
    self.time_series_features = features.get("time_series_features", {})

    # A dictionary of input Tensors. Values have dtype float32 and shape
    # [batch_size, length].
    self.aux_features = features.get("aux_features", {})

    # An int32 Tensor with shape [batch_size]. May be None if mode is
    # tf.estimator.ModeKeys.PREDICT.
    self.labels = labels

    # Optional Tensor; the weights corresponding to self.labels.
    self.weights = features.get("weights")

    # A Python boolean or a scalar boolean Tensor. Indicates whether the model
    # is in training mode for the purpose of graph ops, such as dropout. (Since
    # this might be a Tensor, its value is defined in build()).
    self.is_training = None

    # Global step Tensor.
    self.global_step = None

    # A dictionary of float32 Tensors with shape [batch_size, layer_size]; the
    # outputs of the time series hidden layers.
    self.time_series_hidden_layers = {}

    # A dictionary of float32 Tensors with shape [batch_size, layer_size]; the
    # outputs of the auxiliary hidden layers.
    self.aux_hidden_layers = {}

    # A float32 Tensor with shape [batch_size, layer_size]; the concatenation of
    # outputs from the hidden layers.
    self.pre_logits_concat = None

    # A float32 Tensor with shape [batch_size, output_dim].
    self.logits = None

    # A float32 Tensor with shape [batch_size, output_dim].
    self.predictions = None

    # A float32 Tensor with shape [batch_size]; the cross-entropy losses for the
    # current batch.
    self.batch_losses = None

    # Scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    # No hidden layers.
    self.time_series_hidden_layers = self.time_series_features

  def build_aux_hidden_layers(self):
    """Builds hidden layers for the auxiliary features.

    Inputs:
      self.aux_features

    Outputs:
      self.aux_hidden_layers
    """
    # No hidden layers.
    self.aux_hidden_layers = self.aux_features

  def build_logits(self):
    """Builds the model logits.

    Inputs:
      self.aux_hidden_layers
      self.time_series_hidden_layers

    Outputs:
      self.pre_logits_concat
      self.logits

    Raises:
      ValueError: If self.time_series_hidden_layers and self.aux_hidden_layers
          are both empty.
    """
    # Sort the hidden layers by name because the order of dictionary items is
    # nondeterministic between invocations of Python.
    time_series_hidden_layers = sorted(
        self.time_series_hidden_layers.items(), key=operator.itemgetter(0))
    aux_hidden_layers = sorted(
        self.aux_hidden_layers.items(), key=operator.itemgetter(0))

    hidden_layers = time_series_hidden_layers + aux_hidden_layers
    if not hidden_layers:
      raise ValueError("At least one time series hidden layer or auxiliary "
                       "hidden layer is required.")

    # Concatenate the hidden layers.
    if len(hidden_layers) == 1:
      pre_logits_concat = hidden_layers[0][1]
    else:
      pre_logits_concat = tf.concat([layer[1] for layer in hidden_layers],
                                    axis=1,
                                    name="pre_logits_concat")

    net = tf.cast(pre_logits_concat, tf.float32)
    with tf.name_scope("pre_logits_hidden"):
      for i in range(self.hparams.num_pre_logits_hidden_layers):
        dense_op = tf.keras.layers.Dense(
            units=self.hparams.pre_logits_hidden_layer_size,
            activation=tf.nn.relu,
            dtype=tf.float32,
            name="fully_connected_{}".format(i + 1))
        net = dense_op(net)

        if self.hparams.pre_logits_dropout_rate > 0:
          dropout_op = tf.keras.layers.Dropout(
              self.hparams.pre_logits_dropout_rate)
          net = dropout_op(net, training=self.is_training)

      # Identify the final pre-logits hidden layer as "pre_logits_hidden/final".
      net = tf.identity(net, "final")

    # Create the logits.
    dense_op = tf.keras.layers.Dense(
        units=self.hparams.output_dim,
        dtype=tf.float32,
        name="logits")
    logits = dense_op(net)

    self.pre_logits_concat = pre_logits_concat
    self.logits = tf.cast(logits, tf.float32)

  def build_predictions(self):
    """Builds the output predictions and losses.

    Inputs:
      self.logits

    Outputs:
      self.predictions
    """
    # Use sigmoid activation function for binary classification, or softmax for
    # multi-class classification.
    prediction_fn = (
        tf.sigmoid if self.hparams.output_dim == 1 else tf.nn.softmax)
    predictions = prediction_fn(self.logits, name="predictions")

    self.predictions = predictions

  def build_losses(self):
    """Builds the training losses.

    Inputs:
      self.logits
      self.labels

    Outputs:
      self.batch_losses
      self.total_loss
    """
    if self.hparams.output_dim == 1:
      # Binary classification. Labels are 0 or 1, which can also be interpreted
      # as the probability of the positive class.
      num_classes = 2
      target_probabilities = tf.cast(self.labels, tf.float32)
    else:
      # Multi-class classification. Labels are {0, 1, ..., num_classes - 1},
      # which we convert into one-hot probability distributions.
      num_classes = self.hparams.output_dim
      target_probabilities = tf.one_hot(self.labels, depth=num_classes)

    label_smoothing = self.hparams.get("label_smoothing", 0)
    if label_smoothing > 0:
      # Make target probabilities a weighted mixture of the "hard" probabilities
      # from the dataset and the uniform distribution.
      target_probabilities = (
          target_probabilities * (1 - label_smoothing) +
          label_smoothing / num_classes)

    if self.hparams.output_dim == 1:
      # Binary classification.
      num_classes = 2
      target_probabilities = tf.cast(self.labels, tf.float32)
      
      # Reshape both logits and labels to shape [batch_size, 1]
      logits = tf.reshape(self.logits, [-1, 1])
      target_probabilities = tf.reshape(target_probabilities, [-1, 1])
      
      batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=target_probabilities,
          logits=tf.cast(logits, tf.float32)
      )
    else:
      # Multi-class classification.
      batch_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=tf.cast(target_probabilities, tf.float32), logits=tf.cast(self.logits, tf.float32))

    # Compute the weighted mean cross entropy loss.
    weights = self.weights if self.weights is not None else 1.0
    weights = tf.cast(weights, tf.float32)
    batch_losses = tf.cast(batch_losses, tf.float32)
    total_loss = tf.losses.compute_weighted_loss(
        losses=batch_losses,
        weights=weights,
        reduction=tf.losses.Reduction.MEAN)

    self.batch_losses = batch_losses
    self.total_loss = tf.cast(total_loss, tf.float32)

  def build(self):
    """Creates all ops for training, evaluation or inference."""
    self.global_step = tf.train.get_or_create_global_step()

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      # This is implemented as a placeholder Tensor, rather than a constant, to
      # allow its value to be feedable during training (e.g. to disable dropout
      # when performing in-process validation set evaluation).
      self.is_training = tf.placeholder_with_default(True, [], "is_training")
    else:
      self.is_training = False

    self.build_time_series_hidden_layers()
    self.build_aux_hidden_layers()
    self.build_logits()
    self.build_predictions()

    if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      self.build_losses()
