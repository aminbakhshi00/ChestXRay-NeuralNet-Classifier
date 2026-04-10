import numpy as np
import tensorflow as tf


def build_standard_dataset(ds_inputs, ds_targets, process_path_fn, batch_size, autotune):
    dataset = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))
    dataset = dataset.map(process_path_fn, num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(autotune)


def build_class_balanced_dataset(
    ds_inputs,
    ds_targets,
    process_path_fn,
    batch_size,
    autotune,
    epoch_samples,
    seed=42,
    balance_temperature=0.0,
):
    labels = np.argmax(ds_targets, axis=1)
    num_classes = ds_targets.shape[1]

    class_datasets = []
    class_counts = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if not np.any(class_mask):
            continue

        class_inputs = ds_inputs[class_mask]
        class_targets = ds_targets[class_mask]
        class_counts.append(len(class_inputs))

        class_ds = tf.data.Dataset.from_tensor_slices((class_inputs, class_targets))
        class_ds = class_ds.shuffle(
            buffer_size=max(len(class_inputs), 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        class_ds = class_ds.repeat()
        class_ds = class_ds.map(process_path_fn, num_parallel_calls=autotune)
        class_datasets.append(class_ds)

    if not class_datasets:
        raise ValueError("No class datasets were created from the provided inputs.")

    class_counts = np.array(class_counts, dtype=np.float32)
    if balance_temperature == 0.0:
        weights = np.ones_like(class_counts) / len(class_counts)
    else:
        weights = np.power(class_counts, balance_temperature)
        weights = weights / np.sum(weights)

    weights = weights.tolist()
    dataset = tf.data.Dataset.sample_from_datasets(class_datasets, weights=weights, seed=seed)
    dataset = dataset.take(int(epoch_samples))
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(autotune)
