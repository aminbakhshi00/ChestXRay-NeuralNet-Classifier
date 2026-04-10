import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


class PerClassMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, num_classes, class_names=None, total_epochs=None, max_batches=None):
        super().__init__()
        self.dataset = dataset
        self.num_classes = int(num_classes)
        self.class_names = [str(name) for name in class_names] if class_names is not None else None
        self.total_epochs = None if total_epochs is None else int(total_epochs)
        if max_batches is None:
            self.max_batches = None
        else:
            max_batches = int(max_batches)
            self.max_batches = max_batches if max_batches > 0 else None

    def _get_true_and_pred_labels(self):
        y_true_batches = []
        y_pred_batches = []
        for batch_index, (features, targets) in enumerate(self.dataset):
            if self.max_batches is not None and batch_index >= self.max_batches:
                break
            logits = self.model(features, training=False)
            y_true_batches.append(np.argmax(targets.numpy(), axis=1))
            y_pred_batches.append(np.argmax(logits.numpy(), axis=1))

        if not y_true_batches:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        return np.concatenate(y_true_batches, axis=0), np.concatenate(y_pred_batches, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        planned_epochs = self.params.get("epochs")
        is_last_planned_epoch = planned_epochs is not None and (epoch + 1) == int(planned_epochs)
        is_last_requested_epoch = self.total_epochs is not None and (epoch + 1) == self.total_epochs

        if not (self.model.stop_training or is_last_planned_epoch or is_last_requested_epoch):
            return

        y_true, y_pred = self._get_true_and_pred_labels()
        if y_true.size == 0:
            return

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0,
        )
        macro_f1 = float(np.sum(per_class_f1) / float(self.num_classes))

        for i, score in enumerate(per_class_f1):
            logs[f"f1_class_{i}"] = float(score)

        logs["f1_avg"] = macro_f1

        report_parts = []
        for i, score in enumerate(per_class_f1):
            if self.class_names is not None and i < len(self.class_names):
                label = self.class_names[i]
            else:
                label = str(i)
            report_parts.append(f"{label}:{float(score):.4f}")

        print(" - F1 per class [{}] - f1_avg:{:.4f}".format(", ".join(report_parts), macro_f1))


class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, num_classes=None, class_names=None):
        super().__init__()
        self.dataset = dataset
        self.num_classes = None if num_classes is None else int(num_classes)
        self.class_names = [str(name) for name in class_names] if class_names is not None else None

    def _get_true_and_pred_labels(self):
        y_true_batches = []
        y_pred_batches = []
        for features, targets in self.dataset:
            logits = self.model(features, training=False)
            y_true_batches.append(np.argmax(targets.numpy(), axis=1))
            y_pred_batches.append(np.argmax(logits.numpy(), axis=1))

        if not y_true_batches:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        return np.concatenate(y_true_batches, axis=0), np.concatenate(y_pred_batches, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        y_true, y_pred = self._get_true_and_pred_labels()
        if y_true.size == 0:
            return

        val_f1_macro = float(f1_score(y_true, y_pred, average="macro"))
        logs["val_f1_macro"] = val_f1_macro

    def on_train_end(self, logs=None):
        y_true, y_pred = self._get_true_and_pred_labels()
        if y_true.size == 0:
            return

        if self.num_classes is None:
            num_classes = int(np.max(y_true)) + 1
        else:
            num_classes = self.num_classes

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0,
        )
        macro_f1 = float(np.sum(per_class_f1) / float(num_classes))

        report_parts = []
        for i, score in enumerate(per_class_f1):
            if self.class_names is not None and i < len(self.class_names):
                label = self.class_names[i]
            else:
                label = str(i)
            report_parts.append(f"{label}:{float(score):.4f}")

        print(" - Validation F1 per class [{}] - val_f1_avg:{:.4f}".format(", ".join(report_parts), macro_f1))
