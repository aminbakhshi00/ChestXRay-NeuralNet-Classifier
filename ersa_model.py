import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="ersa")
class add_location_to_patches(tf.keras.layers.Layer):
    def __init__(self, num_tokens, channel_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.channel_dim = channel_dim

    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, self.num_tokens, self.channel_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embedding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_tokens": self.num_tokens,
                "channel_dim": self.channel_dim,
            }
        )
        return config


def _patchify_non_overlapping(x, resized_image_size, patch_size, channels):
    patches_per_side = resized_image_size // patch_size
    num_patches = patches_per_side * patches_per_side
    patch_dim = patch_size * patch_size * channels

    x = tf.keras.layers.Reshape(
        (patches_per_side, patch_size, patches_per_side, patch_size, channels)
    )(x)
    x = tf.keras.layers.Permute((1, 3, 2, 4, 5))(x)
    x = tf.keras.layers.Reshape((num_patches, patch_dim))(x)
    return x, num_patches


def _mixer_block(x, num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout_rate):
    # Token mixing (information exchange across patches)
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Permute((2, 1))(y)
    y = tf.keras.layers.Dense(token_mlp_dim, activation="gelu")(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(num_patches)(y)
    y = tf.keras.layers.Permute((2, 1))(y)
    x = tf.keras.layers.Add()([x, y])

    # Channel mixing (feature learning inside each patch token)
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Dense(channel_mlp_dim, activation="gelu")(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(embed_dim)(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    x = tf.keras.layers.Add()([x, y])
    return x


def _dense_mixer_stack(
    x,
    num_blocks,
    num_patches,
    embed_dim,
    token_mlp_dim,
    channel_mlp_dim,
    dropout_rate,
):
    # DenseNet-style connectivity:
    # each new block sees all previous block outputs.
    block_outputs = [x]

    for block_index in range(num_blocks):
        if len(block_outputs) == 1:
            block_input = block_outputs[0]
        else:
            block_input = tf.keras.layers.Concatenate(axis=-1)(block_outputs)
            block_input = tf.keras.layers.Dense(embed_dim, activation="gelu")(block_input)

        block_output = _mixer_block(
            block_input,
            num_patches=num_patches,
            embed_dim=embed_dim,
            token_mlp_dim=token_mlp_dim,
            channel_mlp_dim=channel_mlp_dim,
            dropout_rate=dropout_rate,
        )
        block_outputs.append(block_output)

    x = tf.keras.layers.Concatenate(axis=-1)(block_outputs)
    x = tf.keras.layers.Dense(embed_dim, activation="gelu")(x)
    return x


def build_dense_patch_mlp(input_dim, num_classes, image_size=300, channels=1):
    resized_image_size = 120
    patch_size = 8
    embed_dim = 128
    token_mlp_dim = 64
    channel_mlp_dim = 256
    num_mixer_blocks = 3
    dropout_rate = 0.15

    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Reshape((image_size, image_size, channels))(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    x = tf.keras.layers.Resizing(resized_image_size, resized_image_size)(x)

    x, num_patches = _patchify_non_overlapping(
        x=x,
        resized_image_size=resized_image_size,
        patch_size=patch_size,
        channels=channels,
    )

    # Shared trainable Dense projection for every patch token
    x = tf.keras.layers.Dense(embed_dim, activation="gelu")(x)
    x = add_location_to_patches(
        num_tokens=num_patches,
        channel_dim=embed_dim,
        name="add_location_to_patches",
    )(x)

    x = _dense_mixer_stack(
        x=x,
        num_blocks=num_mixer_blocks,
        num_patches=num_patches,
        embed_dim=embed_dim,
        token_mlp_dim=token_mlp_dim,
        channel_mlp_dim=channel_mlp_dim,
        dropout_rate=dropout_rate,
    )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=2e-4,
            weight_decay=1e-4,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction="sum_over_batch_size",
        ),
        metrics=["accuracy"],
    )
    return model
