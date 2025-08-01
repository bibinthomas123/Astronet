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

"""Generate predictions for a Threshold Crossing Event using a trained model."""
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

"""Generate predictions for a Threshold Crossing Event using a trained model."""


import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from astronet import models
from astronet.data import preprocess
from astronet.util import estimator_util
from tf_util import config_util
from tf_util import configdict

tf.keras.backend.set_floatx("float32")

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
    help="Name of the model and training configuration. Exactly one of --config_name or --config_json is required.")

parser.add_argument(
    "--config_json",
    type=str,
    help="JSON string or JSON file containing the model and training configuration. Exactly one of --config_name or --config_json is required.")

parser.add_argument("--model_dir", type=str, required=True, help="Directory containing a model checkpoint.")

parser.add_argument("--kepler_data_dir", type=str, required=True, help="Base folder containing Kepler data.")

parser.add_argument("--kepler_id", type=int, required=True, help="Kepler ID of the target star.")

parser.add_argument("--period", type=float, required=True, help="Period of the TCE, in days.")

parser.add_argument("--t0", type=float, required=True, help="Epoch of the TCE.")

parser.add_argument("--duration", type=float, required=True, help="Duration of the TCE, in days.")

parser.add_argument(
    "--output_image_file",
    type=str,
    help="If specified, path to an output image file containing feature plots. Must end in a valid image extension, e.g. png.")

parser.add_argument(
    "--output_prediction_file",
    type=str,
    help="If specified, path to an output text file where prediction will be saved.")


def _process_tce(feature_config):
    """Reads and process the input features of a Threshold Crossing Event."""
    if not {"global_view", "local_view"}.issuperset(feature_config.keys()):
        raise ValueError("Only 'global_view' and 'local_view' features are supported.")

    all_time, all_flux = preprocess.read_light_curve(FLAGS.kepler_id, FLAGS.kepler_data_dir)
    time, flux = preprocess.process_light_curve(all_time, all_flux)
    time, flux = preprocess.phase_fold_and_sort_light_curve(time, flux, FLAGS.period, FLAGS.t0)

    features = {}

    if "global_view" in feature_config:
        global_view = preprocess.global_view(time, flux, FLAGS.period)
        features["global_view"] = np.expand_dims(global_view.astype(np.float32), axis=(0, -1))

    if "local_view" in feature_config:
        local_view = preprocess.local_view(time, flux, FLAGS.period, FLAGS.duration)
        features["local_view"] = np.expand_dims(local_view.astype(np.float32), axis=(0, -1))

    if FLAGS.output_image_file:
        ncols = len(features)
        fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 5), squeeze=False)

        for i, name in enumerate(sorted(features)):
            ax = axes[0][i]
            ax.plot(features[name][0], ".")
            ax.set_title(name)
            ax.set_xlabel("Bucketized Time (days)")
            ax.set_ylabel("Normalized Flux")

        fig.tight_layout()
        fig.savefig(FLAGS.output_image_file, bbox_inches="tight")

    return features


def main(_):
    model_class = models.get_model_class(FLAGS.model)

    assert (FLAGS.config_name is None) != (FLAGS.config_json is None), (
        "Exactly one of --config_name or --config_json is required.")
    config = (
        models.get_model_config(FLAGS.model, FLAGS.config_name)
        if FLAGS.config_name else config_util.parse_json(FLAGS.config_json))
    config = configdict.ConfigDict(config)

    estimator = estimator_util.create_estimator(model_class, config.hparams, model_dir=FLAGS.model_dir)

    features = _process_tce(config.inputs.features)

    def input_fn():
        return tf.data.Dataset.from_tensors({"time_series_features": features})

    predictions = list(estimator.predict(input_fn))
    if not predictions:
        print("No predictions generated!")
        return

    prediction = predictions[0]



    if isinstance(prediction, dict):
        if 'logits' in prediction:
            score = float(tf.sigmoid(prediction['logits']).numpy())
            print("\nPlanet candidate score (0-1) [from logits]:", score)
        elif 'probabilities' in prediction:
            score = float(prediction['probabilities'])
            print("\nPlanet candidate score (0-1) [from probabilities]:", score)
        else:
            print("\nAvailable keys in prediction dict:", prediction.keys())
            score = None

    elif isinstance(prediction, np.ndarray):
        flattened = prediction.flatten()
        score = float(np.max(flattened))  # max chosen as best for TCE detection
        print("\nPlanet candidate score (0-1) [max of output array]:", score)
        
    else:
        print("\nUnexpected prediction type:", type(prediction))
        score = None

    if score is not None:
        with open("prediction.txt", "w+") as f:
            f.write(f"{FLAGS.kepler_id} {score:.6f}\n")

    if FLAGS.output_prediction_file:
        with open(FLAGS.output_prediction_file, "w+") as f:
            f.write(
                f"KIC{FLAGS.kepler_id} = {score:.6f}\n" +
                (f"KIC{FLAGS.kepler_id} = Is a planet candidate\n" if score * 100 > 50
                else f"KIC{FLAGS.kepler_id} = Not a planet candidate\n")
        )


    if score* 100 > 50:
        print(f"\nTCE {FLAGS.kepler_id} is a planet candidate with score {score:.6f} (above 50% threshold).")
    else:
        print(f"\nTCE {FLAGS.kepler_id} is not a planet candidate with score {score:.6f} (below 50% threshold).")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
