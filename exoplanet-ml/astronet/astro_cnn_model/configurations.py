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
  """Base configuration for a CNN + BiLSTM model with a single global view."""
  config = parent_configs.base()

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
          "bilstm_num_layers": 2,
          "bilstm_units": 128,
          "bilstm_dropout_rate": 0.3,
          "bilstm_recurrent_dropout_rate": 0.2,
        #   "bilstm_output_mode": "last",  # 'average', 'max_pool', 'last'
      },
  }

  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 1024
  return config


def local_global():
  """CNN + BiLSTM model with separate local/global views."""
  config = parent_configs.base()

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

  config["hparams"]["time_series_hidden"] = {
      "local_view": {
          # CNN parameters
          "cnn_num_blocks": 3,
          "cnn_block_size": 2,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 3,
          "convolution_padding": "same",
          "pool_size": 3,
          "pool_strides": 2,

          # BiLSTM parameters
          "bilstm_num_layers": 1,
          "bilstm_units": 64,
          "bilstm_dropout_rate": 0.2,
          "bilstm_recurrent_dropout_rate": 0.15,
        #   "bilstm_output_mode": "average",
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
          "bilstm_num_layers": 2,
          "bilstm_units": 128,
          "bilstm_dropout_rate": 0.3,
          "bilstm_recurrent_dropout_rate": 0.2,
        #   "bilstm_output_mode": "average",
      },
  }

  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 256

  # 🚨 Add these
  config["hparams"]["output_dim"] = 1
  config["hparams"]["final_activation"] = "sigmoid"
  config["hparams"]["model_name"] = "cnn_bilstm"

  return config
