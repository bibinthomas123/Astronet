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

"""A model for classifying light curves using a convolutional neural network.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


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
   (convolutional blocks 1)  (convolutional blocks 2)   ...          |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.astro_model import astro_model


class BiLSTMAttentionBlock(tf.keras.layers.Layer):
    """Bidirectional LSTM with attention mechanism."""
    
    def __init__(self, hparams, name="bilstm_attention"):
        super(BiLSTMAttentionBlock, self).__init__(name=name, dtype=tf.float32)
        self.hparams = hparams
        
        # Create LSTM layers with unique names
        self.forward_lstm = tf.keras.layers.LSTM(
            units=hparams.lstm_units,
            return_sequences=True,
            name='forward_lstm',
            dtype=tf.float32  # Explicitly use float32
        )
        
        self.backward_lstm = tf.keras.layers.LSTM(
            units=hparams.lstm_units,
            return_sequences=True,
            name='backward_lstm',
            dtype=tf.float32  # Explicitly use float32
        )
        
        # Attention mechanism
        self.attention_dense = tf.keras.layers.Dense(1, name='attention_score', dtype=tf.float32)
        
    def call(self, inputs):
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # Forward LSTM
        forward_output = self.forward_lstm(inputs)
        
        # Backward LSTM - reverse sequence
        backward_input = tf.reverse(inputs, axis=[1])
        backward_output = self.backward_lstm(backward_input)
        backward_output = tf.reverse(backward_output, axis=[1])
        
        # Concatenate bidirectional outputs
        bilstm_output = tf.concat([forward_output, backward_output], axis=2)
        
        # Compute attention scores
        attention_weights = self.attention_dense(bilstm_output)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # Apply attention
        context_vector = tf.reduce_sum(attention_weights * bilstm_output, axis=1)
        return tf.cast(context_vector, tf.float32)  # Ensure output is float32


class AstroCNNModel(astro_model.AstroModel):
    """A model for classifying light curves using CNN + BiLSTM + Attention."""
    
def _build_cnn_layers(self, inputs, hparams, scope="cnn"):
    """Builds convolutional layers.

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
      A Tensor of shape [batch_size, output_size], where the output size depends
      on the input size, kernel size, number of filters, number of layers,
      convolution padding type and pooling.
    """
    with tf.name_scope(scope):
        # Ensure inputs are Tensor and of correct dtype
        net = tf.convert_to_tensor(inputs)
        net = tf.cast(net, tf.float32)  # Cast to float32 to match checkpoint

        # Expand dims if needed
        if len(net.shape) == 2:
            net = tf.expand_dims(net, -1)  # [batch, length] -> [batch, length, 1]
        elif len(net.shape) != 3:
            raise ValueError("Expected input to have rank 2 or 3. Got shape: {}".format(net.shape))

        # CNN blocks
        for i in range(hparams.cnn_num_blocks):
            num_filters = int(hparams.cnn_initial_num_filters *
                              hparams.cnn_block_filter_factor ** i)
            with tf.name_scope("block_{}".format(i + 1)):
                for j in range(hparams.cnn_block_size):
                    conv_op = tf.keras.layers.Conv1D(
                        filters=num_filters,
                        kernel_size=int(hparams.cnn_kernel_size),
                        padding=hparams.convolution_padding,
                        activation=tf.nn.relu,
                        dtype=tf.float32,  # Use float32 consistently
                        name="conv_{}".format(j + 1))
                    net = conv_op(net)

                if hparams.pool_size > 1:  # pool_size 0 or 1 denotes no pooling
                    pool_op = tf.keras.layers.MaxPool1D(
                        pool_size=int(hparams.pool_size),
                        strides=int(hparams.pool_strides),
                        name="pool")
                    net = pool_op(net)

        # Flatten
        net.shape.assert_has_rank(3)
        net_shape = net.shape.as_list()
        output_dim = net_shape[1] * net_shape[2]
        net = tf.reshape(net, [-1, output_dim], name="flatten")

    return net

def build_time_series_hidden_layers(self):
    """Builds CNN + BiLSTM + Attention layers for the time series features.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    time_series_hidden_layers = {}
    for name, time_series in self.time_series_features.items():
        # Get hyperparameters for this feature
        hparams = self.hparams.time_series_hidden[name]
        
        with tf.name_scope(name + "_hidden"):
            # 1. CNN feature extraction
            cnn_output = self._build_cnn_layers(
                inputs=time_series,
                hparams=hparams,
                scope="cnn"
            )
            
            # Reshape CNN output for BiLSTM
            # Assuming the CNN output is [batch_size, flattened_features]
            # We need to reshape it to [batch_size, sequence_length, features]
            sequence_length = hparams.sequence_length  # You need to add this to hparams
            feature_dim = cnn_output.shape[-1] // sequence_length
            bilstm_input = tf.reshape(cnn_output, [-1, sequence_length, feature_dim])
            
            # 2. BiLSTM + Attention processing
            bilstm_attention = BiLSTMAttentionBlock(
                hparams,
                name=f"{name}_bilstm_attention"
            )
            final_output = bilstm_attention(bilstm_input)
            
            time_series_hidden_layers[name] = final_output

    self.time_series_hidden_layers = time_series_hidden_layers
