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

"""Functions for computing evaluation metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _metric_variable(name, shape, dtype):
  """Creates a Variable in LOCAL_VARIABLES and METRIC_VARIABLES collections."""
  return tf.Variable(
      initial_value=tf.zeros(shape, dtype),
      trainable=False,
      collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
      name=name)


def _build_metrics(labels, predictions, weights, batch_losses, output_dim=1):
  """Builds TensorFlow operations to compute model evaluation metrics.

  Args:
    labels: Tensor with shape [batch_size].
    predictions: Tensor with shape [batch_size] or [batch_size, output_dim].
    weights: Tensor with shape [batch_size].
    batch_losses: Tensor with shape [batch_size].
    output_dim: Dimension of model output

  Returns:
    A dictionary {metric_name: (metric_value, update_op)}.
  """
  with tf.name_scope('metrics_preprocessing'):
    predictions_shape = tf.shape(predictions)
    batch_size = predictions_shape[0]
    num_pred_elements = tf.size(predictions)
    # Ensure labels has correct shape
    labels = tf.reshape(labels, [batch_size])

    binary_classification = output_dim == 1
    if binary_classification:
      # Only reshape if possible, else flatten
      def reshape_pred():
        return tf.reshape(predictions, [batch_size])
      def flatten_pred():
        return tf.reshape(predictions, [-1])
      predictions = tf.cond(tf.equal(num_pred_elements, batch_size), reshape_pred, flatten_pred)
      predicted_labels = tf.cast(tf.greater(predictions, 0.5), tf.int32, name="predicted_labels")
    else:
      # For multi-class, only reshape if possible
      def reshape_pred():
        return tf.reshape(predictions, [batch_size, output_dim])
      def flatten_pred():
        return tf.reshape(predictions, [batch_size, -1])
      predictions = tf.cond(tf.equal(num_pred_elements, batch_size * output_dim), reshape_pred, flatten_pred)
      predicted_labels = tf.argmax(predictions, 1, name="predicted_labels", output_type=tf.int32)

    # Ensure weights match batch size
    weights = tf.reshape(weights, [-1])
    weights = tf.cast(weights, tf.float32)
    weights_size = tf.shape(weights)[0]
    weights = tf.cond(tf.equal(weights_size, batch_size), lambda: weights, lambda: tf.fill([batch_size], 1.0))

  metrics = {}
  with tf.name_scope("metrics"):
    # Total number of examples.
    num_examples = _metric_variable("num_examples", [], tf.float32)
    update_num_examples = tf.assign_add(num_examples, tf.reduce_sum(weights))
    metrics["num_examples"] = (num_examples.read_value(), update_num_examples)

    # Accuracy metrics.
    num_correct = _metric_variable("num_correct", [], tf.float32)
    # Ensure both labels and predicted_labels are [batch_size]
    labels_fixed = labels[:batch_size]
    predicted_labels_fixed = predicted_labels[:batch_size]
    is_correct = tf.equal(labels_fixed, predicted_labels_fixed)
    weighted_is_correct = weights * tf.cast(is_correct, tf.float32)
    update_num_correct = tf.assign_add(num_correct, tf.reduce_sum(weighted_is_correct))
    metrics["accuracy/num_correct"] = (num_correct.read_value(), update_num_correct)
    accuracy = tf.div(num_correct, num_examples, name="accuracy")
    metrics["accuracy/accuracy"] = (accuracy, tf.no_op())

    # Weighted cross-entropy loss.
    metrics["losses/weighted_cross_entropy"] = tf.metrics.mean(
        batch_losses, weights=weights, name="cross_entropy_loss")

    def _count_condition(name, labels_value, predicted_value):
      """Creates a counter for given values of predictions and labels."""
      count = _metric_variable(name, [], tf.float32)
      is_equal = tf.logical_and(
          tf.equal(labels_fixed, labels_value),
          tf.equal(predicted_labels_fixed, predicted_value))
      weighted_is_equal = weights * tf.cast(is_equal, tf.float32)
      update_op = tf.assign_add(count, tf.reduce_sum(weighted_is_equal))
      return count.read_value(), update_op

    # Confusion matrix metrics.
    num_labels = 2 if binary_classification else output_dim
    for gold_label in range(num_labels):
      for pred_label in range(num_labels):
        metric_name = "confusion_matrix/label_{}_pred_{}".format(
            gold_label, pred_label)
        metrics[metric_name] = _count_condition(
            metric_name, labels_value=gold_label, predicted_value=pred_label)

    # AUC metric for binary classification.
    if binary_classification:
      labels = tf.cast(labels, dtype=tf.bool)
      # Ensure predictions and weights are [batch_size]
      pred_size = tf.size(predictions)
      weights_size = tf.size(weights)
      predictions_auc = tf.cond(
          tf.equal(pred_size, batch_size),
          lambda: tf.reshape(predictions, [batch_size]),
          lambda: tf.reshape(predictions, [-1])
      )
      weights_auc = tf.cond(
          tf.equal(weights_size, batch_size),
          lambda: tf.reshape(weights, [batch_size]),
          lambda: tf.reshape(weights, [-1])
      )
      # If still not matching, slice to batch_size
      predictions_auc = predictions_auc[:batch_size]
      weights_auc = weights_auc[:batch_size]
      metrics["auc"] = tf.metrics.auc(
          labels, predictions_auc, weights=weights_auc, num_thresholds=1000)

  return metrics


def create_metric_fn(model):
  """Creates a tuple (metric_fn, metric_fn_inputs) for TPUEstimator."""
  weights = model.weights
  if weights is None:
    weights = tf.ones_like(model.labels, dtype=tf.float32)
  metric_fn_inputs = {
      "labels": model.labels,
      "predictions": model.predictions,
      "weights": weights,
      "batch_losses": model.batch_losses,
  }

  def metric_fn(labels, predictions, weights, batch_losses):
    return _build_metrics(
        labels,
        predictions,
        weights,
        batch_losses,
        output_dim=model.hparams.output_dim)

  return metric_fn, metric_fn_inputs


def create_metrics(model):
  """Creates a dictionary {metric_name: (metric_value, update_op)} for Estimator."""
  metric_fn, metric_fn_inputs = create_metric_fn(model)
  return metric_fn(**metric_fn_inputs)
