# model_arch.py
# Shared model architecture for FogMLS.
# Used by both DL_pipeline.py (training) and IoTsimulation.py (deployment).
# Having one shared file guarantees training and deployment always use
# identical architecture. Never copy this code into other files.
#
# Regularisation changes from baseline:
# - Dropout increased from 0.1 to 0.2 (mild increase)
# - L2 kernel regularisation added to Dense layers (lambda=0.001)
# - These changes target the val_loss gap (2.5-4.2) seen when training
#   on task-only features without capacity. Goal: reduce gap below 2.0
#   while keeping F1 above 0.85 on all fog classes.

import keras
import tensorflow as tf
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.regularizers import L2
from config import MAX_JOBS, MAX_RS, FEATURES_PER_JOB


def make_weighted_loss(class_weights_tensor):
    """
    Returns a weighted categorical crossentropy loss function.
    Class weights are applied per-class before reducing.

    This is the correct approach for multi-output Keras models since
    class_weight in model.fit() is only supported for single-output
    models and raises a ValueError otherwise.

    Args:
        class_weights_tensor: tf.constant of shape [MAX_RS] with one
            weight per class index.
    Returns:
        A loss function compatible with model.compile().
    """
    def weighted_crossentropy(y_true, y_pred):
        ce = -y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
        weighted_ce = ce * class_weights_tensor
        return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))
    return weighted_crossentropy


def build_model(class_weight_dict=None):
    """
    Multi-output MLP for fog resource scheduling.

    Architecture:
    - Input: MAX_JOBS x FEATURES_PER_JOB flat feature vector.
    - Shared backbone: Dense(256, ReLU, L2) -> Dropout(0.2) ->
      Dense(128, ReLU, L2) -> Dropout(0.2).
    - Per-job head: Dense(64, ReLU, L2) -> Dense(MAX_RS, softmax)
      for each of MAX_JOBS positions.

    Regularisation:
    - L2 kernel regularisation (lambda=0.001) on all Dense layers
      penalises large weights, reducing memorisation of training patterns.
    - Dropout(0.2) randomly deactivates 20% of neurons per forward pass,
      forcing the model to learn redundant representations that generalise.
    - These are mild settings chosen to reduce val_loss gap without
      significantly impacting F1 scores on fog classes.

    Args:
        class_weight_dict: Optional dict {class_index: weight}.
    """
    l2_reg = L2(0.001)

    inputs = keras.Input(shape=(MAX_JOBS * FEATURES_PER_JOB,))
    shared = Dense(256, activation='relu',
                   kernel_regularizer=l2_reg)(inputs)
    shared = Dropout(0.2)(shared)
    shared = Dense(128, activation='relu',
                   kernel_regularizer=l2_reg)(shared)
    shared = Dropout(0.2)(shared)

    outputs = []
    output_names = []
    for job in range(MAX_JOBS):
        name = 'job_' + str(job)
        head = Dense(64, activation='relu',
                     kernel_regularizer=l2_reg)(shared)
        out = Dense(MAX_RS, activation='softmax', name=name)(head)
        outputs.append(out)
        output_names.append(name)

    model = keras.Model(inputs=inputs, outputs=outputs)

    if class_weight_dict is not None:
        weights_list = [class_weight_dict.get(i, 1.0) for i in range(MAX_RS)]
        class_weights_tensor = tf.constant(weights_list, dtype=tf.float32)
        loss_fn = make_weighted_loss(class_weights_tensor)
        loss = {name: loss_fn for name in output_names}
        print("[MODEL] Using weighted loss. Weights: " +
              str([round(w, 4) for w in weights_list]))
    else:
        loss = {name: 'categorical_crossentropy' for name in output_names}
        print("[MODEL] Using standard categorical crossentropy.")

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss)

    print("[MODEL] Built with L2+Dropout regularisation:")
    print("  Input=" + str(MAX_JOBS * FEATURES_PER_JOB) +
          ", Jobs=" + str(MAX_JOBS) +
          ", Classes=" + str(MAX_RS) +
          ", Features per job=" + str(FEATURES_PER_JOB))
    print("  L2 lambda=0.001, Dropout=0.2")

    return model, output_names