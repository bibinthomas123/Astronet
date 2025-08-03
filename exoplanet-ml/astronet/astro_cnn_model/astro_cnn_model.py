from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
"""Enhanced model for classifying light curves using CNN + BiLSTM + Attention architecture.

This extends the original Shallue CNN model by adding BiLSTM layers after
the convolutional feature extraction and an attention mechanism to capture 
long-term temporal dependencies in the light curve data.
"""
"""Enhanced model for classifying light curves using CNN + BiLSTM + Attention architecture.

This extends the original Shallue CNN model by adding BiLSTM layers after
the convolutional feature extraction and an attention mechanism to capture 
long-term temporal dependencies in the light curve data.
"""


import tensorflow as tf

from astronet.astro_model import astro_model


class AstroCNNModel(astro_model.AstroModel):
  """Enhanced model for classifying light curves using CNN + BiLSTM + Attention architecture."""

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
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      net = inputs
      if net.shape.rank == 2:
        net = tf.expand_dims(net, -1)  # [batch, length] -> [batch, length, 1]
      if net.shape.rank != 3:
        raise ValueError(
            "Expected inputs to have rank 2 or 3. Got: {}".format(inputs))
      
      for i in range(hparams.cnn_num_blocks):
        num_filters = int(hparams.cnn_initial_num_filters *
                          hparams.cnn_block_filter_factor**i)
        block_scope = "block_{}".format(i + 1)
        
        with tf.variable_scope(block_scope):
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
      A Tensor of shape [batch_size, sequence_length, 2*bilstm_units] after BiLSTM processing.
      Note: This returns sequences for attention mechanism.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      net = inputs
      
      # Ensure we have the right input shape
      net.shape.assert_has_rank(3)
      
      # Add BiLSTM layers
      for i in range(hparams.bilstm_num_layers):
        layer_scope = "bilstm_layer_{}".format(i + 1)
        with tf.variable_scope(layer_scope):
          lstm_units = int(hparams.bilstm_units)
          
          # Apply dropout if specified
          if hasattr(hparams, 'bilstm_dropout_rate') and hparams.bilstm_dropout_rate > 0:
            net = tf.layers.dropout(
                net, 
                rate=hparams.bilstm_dropout_rate,
                training=self.is_training,
                name="dropout_{}".format(i + 1)
            )
          
          # For attention, we need the last layer to return sequences
          return_sequences = True
          
          # Use tf.nn.dynamic_rnn instead of Keras layers for better TF 1.x compatibility
          with tf.variable_scope("forward"):
            fw_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=lstm_units,
                state_is_tuple=True,
                name="fw_lstm_cell_{}".format(i + 1)
            )
            if hasattr(hparams, 'bilstm_dropout_rate') and hparams.bilstm_dropout_rate > 0:
              fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                  fw_cell,
                  output_keep_prob=1.0 - hparams.bilstm_dropout_rate,
                  state_keep_prob=1.0 - getattr(hparams, 'bilstm_recurrent_dropout_rate', 0.0)
              )
          
          with tf.variable_scope("backward"):
            bw_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=lstm_units,
                state_is_tuple=True,
                name="bw_lstm_cell_{}".format(i + 1)
            )
            if hasattr(hparams, 'bilstm_dropout_rate') and hparams.bilstm_dropout_rate > 0:
              bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                  bw_cell,
                  output_keep_prob=1.0 - hparams.bilstm_dropout_rate,
                  state_keep_prob=1.0 - getattr(hparams, 'bilstm_recurrent_dropout_rate', 0.0)
              )
          
          # Use bidirectional_dynamic_rnn
          outputs, states = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=fw_cell,
              cell_bw=bw_cell,
              inputs=net,
              dtype=tf.float32,
              scope="birnn_{}".format(i + 1)
          )
          
          # Concatenate forward and backward outputs
          net = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, 2*lstm_units]
      
      return net  # Returns [batch_size, sequence_length, 2*bilstm_units]

  def _apply_attention_layer(self, inputs, scope="attention"):
    """Applies a simple attention mechanism on BiLSTM outputs.
    
    Args:
      inputs: A Tensor of shape [batch_size, sequence_length, features]
      scope: Prefix for operation names.
      
    Returns:
      A Tensor of shape [batch_size, features] after applying attention.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      # inputs: [batch_size, sequence_length, features]
      
      # Compute attention scores using tf.layers.dense for TF 1.x compatibility
      scores = tf.layers.dense(
          inputs,
          units=1,
          activation=tf.nn.tanh,
          name="attention_score"
      )  # [batch_size, sequence_length, 1]
      
      # Apply softmax to get attention weights
      attention_weights = tf.nn.softmax(scores, axis=1)  # [batch_size, sequence_length, 1]
      
      # Apply attention weights to the input
      context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)  # [batch_size, features]
      
      # Ensure the output has the correct shape
      context_vector = tf.ensure_shape(context_vector, [None, inputs.shape[-1]])
      
      return context_vector
    
  # def _apply_attention_layer(self, inputs, scope="attention"):
  #   """Applies a simple attention mechanism on BiLSTM outputs.
    
  #   Args:
  #     inputs: A Tensor of shape [batch_size, sequence_length, features]
  #     scope: Prefix for operation names.
      
  #   Returns:
  #     A Tensor of shape [batch_size, features] after applying attention.
  #   """
  #   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
  #     # inputs: [batch_size, sequence_length, features]
      
  #     # Compute attention scores
  #     # Use a dense layer to compute attention scores for each timestep
  #     attention_dense = tf.keras.layers.Dense(1, activation='tanh', name="attention_score")
  #     scores = attention_dense(inputs)  # [batch_size, sequence_length, 1]
      
  #     # Apply softmax to get attention weights
  #     attention_weights = tf.nn.softmax(scores, axis=1)  # [batch_size, sequence_length, 1]
      
  #     # Apply attention weights to the input
  #     context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)  # [batch_size, features]
      
  #     # Ensure the output has the correct shape
  #     context_vector = tf.ensure_shape(context_vector, [None, inputs.shape[-1]])
      
  #     return context_vector

  def _build_cnn_bilstm_layers(self, inputs, hparams, scope="cnn_bilstm"):
    """Builds combined CNN + BiLSTM + Attention layers.

    Args:
      inputs: A Tensor of shape [batch_size, length] or
        [batch_size, length, ndims].
      hparams: Object containing both CNN and BiLSTM hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, output_size].
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      # First apply CNN layers
      cnn_output = self._build_cnn_layers(inputs, hparams, scope="cnn_part")
      
      # Then apply BiLSTM layers (returns sequences for attention)
      bilstm_output = self._build_bilstm_layers(cnn_output, hparams, scope="bilstm_part")
      
      # Apply attention mechanism on BiLSTM output
      attention_output = self._apply_attention_layer(bilstm_output, scope="attention_part")
      
      return attention_output

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features using CNN + BiLSTM + Attention.

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