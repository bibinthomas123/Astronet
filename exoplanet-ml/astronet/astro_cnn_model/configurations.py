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

"""Configurations for model building, training and evaluation.

Available configurations:
  * base: One time series feature per input example. Default is "global_view".
  * local_global: Two time series features per input example.
      - A "global" view of the entire orbital period.
      - A "local" zoomed-in view of the transit event.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astronet.astro_model import configurations as parent_configs


def base():
  """Base configuration for a CNN model with a single global view."""
  config = parent_configs.base()

  # Add configuration for the CNN + BiLSTM + Attention layers
  config["hparams"]["time_series_hidden"] = {
      "global_view": {
          # CNN parameters
          "cnn_num_blocks": 5,
          "cnn_block_size": 2,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 5,
          "convolution_padding": "same",
          "pool_size": 5,
          "pool_strides": 2,
          
          # BiLSTM parameters
          "lstm_units": 64,
          "sequence_length": 64,  # This should be set based on your input length after CNN processing
          "lstm_dropout": 0.1,
      },
  }
  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 1024
  return config


def local_global():
  """Base configuration for a CNN model with separate local/global views."""
  config = parent_configs.base()

  # Override the model features to be local_view and global_view time series.
  config["inputs"]["features"] = {
      "local_view": {
          "length": 201,
          "is_time_series": True,
          "subcomponents": []
      },
      "global_view": {
          "length": 2001,
          "is_time_series": True,
          "subcomponents": []
      },
  }

  # Add configurations for the convolutional layers of time series features.
  config["hparams"]["time_series_hidden"] = {
      "local_view": {
          # CNN parameters
          "cnn_num_blocks": 2,
          "cnn_block_size": 2,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 5,
          "convolution_padding": "same",
          "pool_size": 7,
          "pool_strides": 2,
          
          # BiLSTM parameters
          "lstm_units": 32,  # Smaller for local view
          "sequence_length": 32,  # Smaller sequence length for local view
          "lstm_dropout": 0.1,
      },
      "global_view": {
          # CNN parameters
          "cnn_num_blocks": 5,
          "cnn_block_size": 2,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 5,
          "convolution_padding": "same",
          "pool_size": 5,
          "pool_strides": 2,
          
          # BiLSTM parameters
          "lstm_units": 64,  # Larger for global view
          "sequence_length": 64,  # Larger sequence length for global view
          "lstm_dropout": 0.1,
      },
  }
  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 512
  return config
