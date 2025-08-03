Input Light Curve Data
         ↓
    [batch_size, length]
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CNN FEATURE EXTRACTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: [batch_size, length] → [batch_size, length, 1] (expand dims)       │
│                                                                             │
│  Block 1: filters = 16                                                     │
│  ├── Conv1D(kernel=5, filters=16, activation=ReLU, padding=same)           │
│  ├── Conv1D(kernel=5, filters=16, activation=ReLU, padding=same)           │
│  └── MaxPool1D(pool_size=5, strides=2)                                     │
│                                                                             │
│  Block 2: filters = 32                                                     │
│  ├── Conv1D(kernel=5, filters=32, activation=ReLU, padding=same)           │
│  ├── Conv1D(kernel=5, filters=32, activation=ReLU, padding=same)           │
│  └── MaxPool1D(pool_size=5, strides=2)                                     │
│                                                                             │
│  Block 3: filters = 64                                                     │
│  ├── Conv1D(kernel=5, filters=64, activation=ReLU, padding=same)           │
│  ├── Conv1D(kernel=5, filters=64, activation=ReLU, padding=same)           │
│  └── MaxPool1D(pool_size=5, strides=2)                                     │
│                                                                             │
│  Block 4: filters = 128                                                    │
│  ├── Conv1D(kernel=5, filters=128, activation=ReLU, padding=same)          │
│  ├── Conv1D(kernel=5, filters=128, activation=ReLU, padding=same)          │
│  └── MaxPool1D(pool_size=5, strides=2)                                     │
│                                                                             │
│  Block 5: filters = 256                                                    │
│  ├── Conv1D(kernel=5, filters=256, activation=ReLU, padding=same)          │
│  ├── Conv1D(kernel=5, filters=256, activation=ReLU, padding=same)          │
│  └── MaxPool1D(pool_size=5, strides=2)                                     │
│                                                                             │
│  Output: [batch_size, reduced_sequence_length, 256]                        │
└─────────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BIDIRECTIONAL LSTM LAYERS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: [batch_size, sequence_length, 256]                                 │
│                                                                             │
│  BiLSTM Layer 1: 128 units                                                 │
│  ├── Forward LSTM: 128 units                                               │
│  ├── Backward LSTM: 128 units                                              │
│  ├── Dropout: 0.3 (if training)                                            │
│  ├── Recurrent Dropout: 0.2                                                │
│  └── Concatenate → [batch_size, sequence_length, 256]                      │
│                                                                             │
│  BiLSTM Layer 2: 128 units                                                 │
│  ├── Forward LSTM: 128 units                                               │
│  ├── Backward LSTM: 128 units                                              │
│  ├── Dropout: 0.3 (if training)                                            │
│  ├── Recurrent Dropout: 0.2                                                │
│  └── Concatenate → [batch_size, sequence_length, 256]                      │
│                                                                             │
│  Output: [batch_size, sequence_length, 256] (returns sequences)            │
└─────────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ATTENTION MECHANISM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: [batch_size, sequence_length, 256]                                 │
│                                                                             │
│  Attention Score Computation:                                              │
│  ├── Dense(1, activation=tanh) → [batch_size, sequence_length, 1]          │
│  ├── Softmax(axis=1) → attention_weights [batch_size, sequence_length, 1]  │
│  └── Weighted Sum → context_vector [batch_size, 256]                       │
│                                                                             │
│  Formula: context = Σ(attention_weights[i] * hidden_states[i])             │
│                                                                             │
│  Output: [batch_size, 256] (context vector)                                │
└─────────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE CONCATENATION & PROCESSING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Multiple Time Series Features (if any):                                   │
│  ├── global_view: [batch_size, 256]                                        │
│  ├── local_view: [batch_size, 256] (if configured)                         │
│  └── Concatenate → [batch_size, total_features]                            │
│                                                                             │
│  Auxiliary Features (if any):                                              │
│  ├── period, duration, etc.: [batch_size, aux_features]                    │
│  └── Concatenate with time series features                                 │
│                                                                             │
│  Output: pre_logits_concat [batch_size, total_feature_size]                │
└─────────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FULLY CONNECTED LAYERS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: [batch_size, total_feature_size]                                   │
│                                                                             │
│  Pre-Logits Hidden Layers (4 layers):                                      │
│  ├── Dense(1024, activation=ReLU)                                          │
│  ├── Dropout(0.2) (if training)                                            │
│  ├── Dense(1024, activation=ReLU)                                          │
│  ├── Dropout(0.2) (if training)                                            │
│  ├── Dense(1024, activation=ReLU)                                          │
│  ├── Dropout(0.2) (if training)                                            │
│  ├── Dense(1024, activation=ReLU)                                          │
│  └── Dropout(0.2) (if training)                                            │
│                                                                             │
│  Output: [batch_size, 1024]                                                │
└─────────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Logits Layer:                                                             │
│  └── Dense(1) → [batch_size, 1] (raw logits)                              │
│                                                                             │
│  Predictions:                                                              │
│  └── Sigmoid(logits) → [batch_size, 1] (probabilities 0-1)                │
│                                                                             │
│  Loss (during training):                                                   │
│  └── Binary Cross-Entropy with Label Smoothing                            │
└─────────────────────────────────────────────────────────────────────────────┘
         ↓
    Final Prediction
   (Planet/Not Planet)