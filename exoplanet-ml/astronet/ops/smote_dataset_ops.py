"""Dataset operations with SMOTE balancing for AstroNet."""

import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors




def smote_balance_dataset(dataset, 
                         batch_size,
                         positive_feature_name="global_view",
                         label_feature_name="av_training_set",
                         num_neighbors=5):
    """Applies SMOTE balancing to a TensorFlow dataset.
    
    Args:
        dataset: A tf.data.Dataset containing TCE examples.
        batch_size: The batch size to use for processing.
        positive_feature_name: Name of the feature containing light curve data.
        label_feature_name: Name of the label feature.
        num_neighbors: Number of nearest neighbors to use for SMOTE.
        
    Returns:
        A balanced dataset with synthetic minority examples.
    """
    # Convert to numpy for preprocessing
    examples = []
    labels = []
    
    for example in dataset:
        feature = example[positive_feature_name]
        label = example[label_feature_name]
        examples.append(feature.numpy())
        labels.append(label.numpy())
    
    examples = np.array(examples)
    labels = np.array(labels)
    
    # Separate majority and minority classes
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    if len(positive_indices) == 0:
        return dataset
        
    # Get positive examples
    positive_examples = examples[positive_indices]
    
    # Calculate number of synthetic examples needed
    num_synthetic = len(negative_indices) - len(positive_indices)
    
    # Find k nearest neighbors for each positive example
    knn = NearestNeighbors(n_neighbors=num_neighbors + 1)  # +1 because first neighbor is self
    knn.fit(positive_examples.reshape(len(positive_examples), -1))
    
    synthetic_examples = []
    for i in range(len(positive_examples)):
        nn_indices = knn.kneighbors(positive_examples[i].reshape(1, -1), return_distance=False)[0][1:]
        num_samples = min(num_neighbors, (num_synthetic + num_neighbors - 1) // len(positive_indices))
        
        for _ in range(num_samples):
            nn_idx = np.random.choice(nn_indices)
            alpha = np.random.random()
            
            # Generate synthetic example
            synthetic = positive_examples[i] + alpha * (positive_examples[nn_idx] - positive_examples[i])
            synthetic_examples.append(synthetic)
            
        num_synthetic -= num_samples
        if num_synthetic <= 0:
            break
    
    if synthetic_examples:
        synthetic_examples = np.array(synthetic_examples)
        
        # Create dataset with synthetic examples
        synthetic_dataset = tf.data.Dataset.from_tensor_slices({
            positive_feature_name: synthetic_examples,
            label_feature_name: np.ones(len(synthetic_examples), dtype=np.int32)
        })
        
        # Combine original and synthetic datasets
        balanced_dataset = dataset.concatenate(synthetic_dataset)
        
        # Shuffle and batch
        return balanced_dataset.shuffle(buffer_size=1000).batch(batch_size)
    
    return dataset.batch(batch_size)


def create_input_fn(file_pattern,
                   input_config,
                   mode,
                   batch_size=32,
                   shuffle_values_buffer=0,
                   repeat=1,
                   apply_smote=True):
    """Creates an input_fn that reads a dataset from sharded TFRecord files.
    
    Args:
        file_pattern: File pattern matching input TFRecord files.
        input_config: ConfigDict containing feature and label specifications.
        mode: A tf.estimator.ModeKeys.
        batch_size: The batch size to use.
        shuffle_values_buffer: If > 0, shuffle examples using a buffer of this size.
        repeat: The number of times to repeat the dataset.
        apply_smote: Whether to apply SMOTE balancing.
        
    Returns:
        A function that returns a tf.data.Dataset object.
    """
    def input_fn(params=None):
        # Create the dataset
        dataset = tf.data.Dataset.list_files(file_pattern)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Parse examples
        dataset = dataset.map(
            lambda x: tf.io.parse_single_example(x, input_config.features),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if mode == tf.estimator.ModeKeys.TRAIN and apply_smote:
            # Apply SMOTE balancing for training
            dataset = smote_balance_dataset(
                dataset,
                batch_size=batch_size,
                positive_feature_name="global_view",
                label_feature_name="av_training_set")
        else:
            # For evaluation and prediction, just batch the data
            if shuffle_values_buffer > 0:
                dataset = dataset.shuffle(shuffle_values_buffer)
            if repeat != 1:
                dataset = dataset.repeat(repeat)
            dataset = dataset.batch(batch_size)
        
        return dataset
    
    return input_fn
