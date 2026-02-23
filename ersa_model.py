import tensorflow as tf


def _mixer_block(x, num_tokens, channel_dim, token_mlp_dim, channel_mlp_dim, dropout_rate):
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
    y = tf.keras.layers.Permute((2, 1))(y)
    y = tf.keras.layers.Dense(token_mlp_dim, activation="gelu")(y)
    # y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(num_tokens)(y)
    y = tf.keras.layers.Permute((2, 1))(y)
    x = tf.keras.layers.Add()([x, y])

    y = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
    gate = tf.keras.layers.Dense(channel_mlp_dim, activation="sigmoid")(y)
    value = tf.keras.layers.Dense(channel_mlp_dim, activation="gelu")(y)
    y = tf.keras.layers.Multiply()([gate, value])
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(channel_dim)(y)
    # y = tf.keras.layers.Dropout(dropout_rate)(y)
    return tf.keras.layers.Add()([x, y])


def build_dense_patch_mlp(input_dim, num_classes, image_size=300, channels=1):
    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Reshape((image_size, image_size, channels))(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    x = tf.keras.layers.Resizing(120, 120)(x)

    patch_size = 8
    patches_per_side = 120 // patch_size
    num_tokens = patches_per_side * patches_per_side
    patch_dim = patch_size * patch_size * channels

    x = tf.keras.layers.Reshape((patches_per_side, patch_size, patches_per_side, patch_size, channels))(x)
    x = tf.keras.layers.Permute((1, 3, 2, 4, 5))(x)
    x = tf.keras.layers.Reshape((num_tokens, patch_dim))(x)

    channel_dim = 256
    x = tf.keras.layers.Dense(channel_dim, activation="gelu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    for _ in range(4):
        x = _mixer_block(
            x,
            num_tokens=num_tokens,
            channel_dim=channel_dim,
            token_mlp_dim=128,
            channel_mlp_dim=512,
            dropout_rate=0.1,
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
    x_mean = tf.keras.layers.GlobalAveragePooling1D()(x)
    x_max = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([x_mean, x_max])

    branch_a = tf.keras.layers.Dense(384, activation="gelu")(x)
    branch_a = tf.keras.layers.Dropout(0.1)(branch_a)
    branch_b = tf.keras.layers.Dense(384, activation="relu")(x)
    branch_b = tf.keras.layers.Dropout(0.1)(branch_b)

    x = tf.keras.layers.Concatenate()([branch_a, branch_b])
    x = tf.keras.layers.Add()([x, tf.keras.layers.Dense(768)(tf.keras.layers.Concatenate()([x_mean, x_max]))])
    x = tf.keras.layers.Dense(192, activation="gelu")(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model
