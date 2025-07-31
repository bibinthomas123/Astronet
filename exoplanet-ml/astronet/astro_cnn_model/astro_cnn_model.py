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

"""Enhanced model for classifying light curves using CNN + BiLSTM architecture.

This extends the original Shallue CNN model by adding BiLSTM layers after
the convolutional feature extraction to capture long-term temporal dependencies
in the light curve data.

The enhanced architecture is:

                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
      (CNN + BiLSTM blocks 1)    (CNN + BiLSTM blocks 2)   ...      |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.astro_model import astro_model


class AstroCNNModel(astro_model.AstroModel):
  """Enhanced model for classifying light curves using CNN + BiLSTM architecture."""

  def _build_cnn_layers(self, inputs, hparams, scope="cnn"):
    """Builds convolutional layers (same as original).

    The layers are defined by convolutional blocks with pooling between blocks
    (but not within blocks). Within a block, all layers have the same number of
    filters, which is a constant multiple of the number of filters in the
    previous block. The kernel size is fixed throughout.

    Args:
      inputs: A Tensor of shape [batch_size, length] or
        [batch_size, length, ndims].
      hparams: Object containing CNN hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, sequence_length, features] for BiLSTM input
      or [batch_size, output_size] if flattened.
    """
    with tf.name_scope(scope):
      net = inputs
      if net.shape.rank == 2:
        net = tf.expand_dims(net, -1)  # [batch, length] -> [batch, length, 1]
      if net.shape.rank != 3:
        raise ValueError(
            "Expected inputs to have rank 2 or 3. Got: {}".format(inputs))
      
      for i in range(hparams.cnn_num_blocks):
        num_filters = int(hparams.cnn_initial_num_filters *
                          hparams.cnn_block_filter_factor**i)
        with tf.name_scope("block_{}".format(i + 1)):
          for j in range(hparams.cnn_block_size):
            conv_op = tf.keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=int(hparams.cnn_kernel_size),
                padding=hparams.convolution_padding,
                activation=tf.nn.relu,
                name="conv_{}".format(j + 1))
            net = conv_op(net)

          if hparams.pool_size > 1:  # pool_size 0 or 1 denotes no pooling
            pool_op = tf.keras.layers.MaxPool1D(
                pool_size=int(hparams.pool_size),
                strides=int(hparams.pool_strides),
                name="pool")
            net = pool_op(net)

      return net  # Return 3D tensor for BiLSTM processing

  def _build_bilstm_layers(self, inputs, hparams, scope="bilstm"):
    """Builds bidirectional LSTM layers after CNN feature extraction.

    Args:
      inputs: A Tensor of shape [batch_size, sequence_length, features] from CNN.
      hparams: Object containing BiLSTM hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, output_size] after BiLSTM processing.
    """
    with tf.name_scope(scope):
      net = inputs
      
      # Ensure we have the right input shape
      net.shape.assert_has_rank(3)
      
      # Add BiLSTM layers
      for i in range(hparams.bilstm_num_layers):
        with tf.name_scope("bilstm_layer_{}".format(i + 1)):
          # Create forward and backward LSTM cells
          lstm_units = int(hparams.bilstm_units)
          
          # Apply dropout if specified
          if hasattr(hparams, 'bilstm_dropout_rate') and hparams.bilstm_dropout_rate > 0:
            net = tf.keras.layers.Dropout(rate=hparams.bilstm_dropout_rate)(net)
          
          # Bidirectional LSTM layer
          bilstm_layer = tf.keras.layers.Bidirectional(
              tf.keras.layers.LSTM(
                  units=lstm_units,
                  return_sequences=(i < hparams.bilstm_num_layers - 1),  # Return sequences for all but last layer
                  dropout=getattr(hparams, 'bilstm_recurrent_dropout_rate', 0.0),
                  recurrent_dropout=getattr(hparams, 'bilstm_recurrent_dropout_rate', 0.0),
                  name="lstm_{}".format(i + 1)
              ),
              name="bidirectional_{}".format(i + 1)
          )
          net = bilstm_layer(net)
      
      # If the last layer returned sequences, we need to handle the output
      if net.shape.rank == 3:
        # Option 1: Use the last timestep output
        if hasattr(hparams, 'bilstm_output_mode') and hparams.bilstm_output_mode == 'last':
          net = net[:, -1, :]  # Take last timestep
        # Option 2: Global max pooling
        elif hasattr(hparams, 'bilstm_output_mode') and hparams.bilstm_output_mode == 'max_pool':
          net = tf.keras.layers.GlobalMaxPooling1D()(net)
        # Option 3: Global average pooling (default)
        else:
          net = tf.keras.layers.GlobalAveragePooling1D()(net)
      
      return net

  def _build_cnn_bilstm_layers(self, inputs, hparams, scope="cnn_bilstm"):
    """Builds combined CNN + BiLSTM layers.

    Args:
      inputs: A Tensor of shape [batch_size, length] or
        [batch_size, length, ndims].
      hparams: Object containing both CNN and BiLSTM hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, output_size].
    """
    with tf.name_scope(scope):
      # First apply CNN layers
      cnn_output = self._build_cnn_layers(inputs, hparams, scope="cnn_part")
      
      # Then apply BiLSTM layers
      bilstm_output = self._build_bilstm_layers(cnn_output, hparams, scope="bilstm_part")
      
      return bilstm_output

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features using CNN + BiLSTM.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    time_series_hidden_layers = {}
    for name, time_series in self.time_series_features.items():
      time_series_hidden_layers[name] = self._build_cnn_bilstm_layers(
          inputs=time_series,
          hparams=self.hparams.time_series_hidden[name],
          scope=name + "_hidden")

    self.time_series_hidden_layers = time_series_hidden_layers

