#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from ersa_model import build_dense_patch_mlp
from ersa_callbacks import PerClassMacroF1Callback, ValidationMacroF1Callback
from ersa_dataloader import build_standard_dataset

#------------------------------------------------------------------------------------------------------------------

'''
LAST UPDATE 10/20/2021 LSDR
last update 10/21/2021 lsdr
02/14/2022 am LSDR CHECK CONSISTENCY
02/14/2022 pm LSDR Change result for results

'''
#------------------------------------------------------------------------------------------------------------------
## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 50
BATCH_SIZE = 32
NUM_SUBMODELS = 4
F1_EVAL_MAX_BATCHES = 200
VALIDATION_SPLIT = 0.15
SPLIT_RANDOM_STATE = 42
EARLY_STOP_PATIENCE = 7
LR_PATIENCE = 5

## Image processing
CHANNELS = 1
IMAGE_SIZE = 300

NICKNAME = 'Ersa'

TRAIN_RANDOM_FLIP = tf.keras.layers.RandomFlip("horizontal")
TRAIN_RANDOM_ROTATION = tf.keras.layers.RandomRotation(0.02, fill_mode="reflect")
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''


    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = xdf_data['target'].apply(x)

        final_target = to_categorical(list(final_target))

        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in  (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))

        xdepth = len(class_names)

        final_target = tf.one_hot(target, xdepth)

        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names
#------------------------------------------------------------------------------------------------------------------

def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])

    img = tf.reshape(img, [-1])

    return img, label
#------------------------------------------------------------------------------------------------------------------

def process_path_train(feature, target):
    '''
          train-time path processing with augmentation only for training data
    '''

    img, label = process_path(feature, target)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    img = TRAIN_RANDOM_FLIP(img, training=True)
    img = TRAIN_RANDOM_ROTATION(img, training=True)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.reshape(img, [-1])
    return img, label
#------------------------------------------------------------------------------------------------------------------

def get_target(dset_df, num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(dset_df['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes, dtype=np.float32)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:], dtype=np.float32)

    return y_target
#------------------------------------------------------------------------------------------------------------------


def read_data(dset_df, num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + dset_df['id'])
    ds_targets = get_target(dset_df, num_classes)
    return build_standard_dataset(ds_inputs, ds_targets, process_path, BATCH_SIZE, AUTOTUNE)
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    model = build_dense_patch_mlp(
        input_dim=INPUTS_r,
        num_classes=OUTPUTS_a,
        image_size=IMAGE_SIZE,
        channels=CHANNELS,
    )
    return model
#------------------------------------------------------------------------------------------------------------------

def read_data_train_standard(dset_df, num_classes, seed):
    '''
          reads one train subset and applies train-only augmentation
    '''

    ds_inputs = np.array(DATA_DIR + dset_df['id'])
    ds_targets = get_target(dset_df, num_classes)
    train_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))
    train_ds = train_ds.shuffle(
        buffer_size=max(len(ds_inputs), 1),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    train_ds = train_ds.map(process_path_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    return train_ds.prefetch(AUTOTUNE)
#------------------------------------------------------------------------------------------------------------------

def build_submodel_train_splits(
    xdf_train,
    class_names,
    num_submodels=NUM_SUBMODELS,
    anchor_class='class5',
    seed=SPLIT_RANDOM_STATE,
):
    '''
          builds per-submodel train data using:
          - disjoint partitions for non-anchor classes
          - top-up sampling to match anchor class count
    '''

    class_name_list = [str(name) for name in class_names]
    if anchor_class not in class_name_list:
        raise ValueError("Required anchor class '{}' was not found.".format(anchor_class))

    class_frames = {}
    for class_name in class_name_list:
        class_df = xdf_train[xdf_train['target'] == class_name].copy().reset_index(drop=True)
        if class_df.empty:
            raise ValueError("Class '{}' has no training rows.".format(class_name))
        class_frames[class_name] = class_df

    anchor_df = class_frames[anchor_class].copy().reset_index(drop=True)
    target_count_per_class = len(anchor_df)
    print("Anchor class '{}' rows per submodel: {}".format(anchor_class, target_count_per_class))

    non_anchor_partitions = {}
    for class_name in class_name_list:
        if class_name == anchor_class:
            continue
        shuffled_df = class_frames[class_name].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        index_chunks = np.array_split(np.arange(len(shuffled_df)), num_submodels)
        chunks = [shuffled_df.iloc[idx_chunk].reset_index(drop=True) for idx_chunk in index_chunks]
        non_anchor_partitions[class_name] = chunks

        covered_ids = set()
        for chunk in chunks:
            covered_ids.update(chunk['id'].tolist())
        original_ids = set(shuffled_df['id'].tolist())
        if covered_ids != original_ids:
            raise ValueError("Coverage check failed for class '{}' partitions.".format(class_name))

    submodel_splits = []
    for model_index in range(num_submodels):
        split_parts = [anchor_df.copy()]

        for class_offset, class_name in enumerate(class_name_list):
            if class_name == anchor_class:
                continue

            base_chunk = non_anchor_partitions[class_name][model_index].copy().reset_index(drop=True)
            needed = target_count_per_class - len(base_chunk)

            if needed > 0:
                top_up = class_frames[class_name].sample(
                    n=needed,
                    replace=True,
                    random_state=seed + (model_index + 1) * 100 + class_offset,
                ).reset_index(drop=True)
                class_chunk = pd.concat([base_chunk, top_up], ignore_index=True)
            elif needed < 0:
                class_chunk = base_chunk.sample(
                    n=target_count_per_class,
                    random_state=seed + (model_index + 1) * 100 + class_offset,
                ).reset_index(drop=True)
            else:
                class_chunk = base_chunk

            split_parts.append(class_chunk)

        split_df = pd.concat(split_parts, ignore_index=True)
        split_df = split_df.sample(frac=1.0, random_state=seed + model_index).reset_index(drop=True)

        split_counts = split_df['target'].value_counts()
        split_report = ", ".join(
            "{}:{}".format(class_name, int(split_counts.get(class_name, 0)))
            for class_name in class_name_list
        )
        print("Submodel {} class counts -> {}".format(model_index + 1, split_report))

        for class_name in class_name_list:
            if int(split_counts.get(class_name, 0)) != target_count_per_class:
                raise ValueError(
                    "Class '{}' in submodel {} is not balanced to anchor count.".format(
                        class_name, model_index + 1
                    )
                )

        submodel_splits.append(split_df)

    return submodel_splits
#------------------------------------------------------------------------------------------------------------------

def train_single_submodel(model_index, train_ds, val_ds):
    '''
        train one branch model and save best checkpoint
    '''

    checkpoint_path = 'tmp_model_{}_{}.keras'.format(NICKNAME, model_index + 1)
    check_point = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_f1_macro',
        mode='max',
        save_best_only=True,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_macro',
        mode='max',
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_macro',
        mode='max',
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=1e-6,
    )
    val_f1_callback = ValidationMacroF1Callback(val_ds, OUTPUTS_a, class_names=class_names)
    f1_callback = PerClassMacroF1Callback(
        train_ds,
        OUTPUTS_a,
        class_names=class_names,
        max_batches=F1_EVAL_MAX_BATCHES,
    )
    final_model = model_definition()

    final_model.fit(
        train_ds,
        epochs=n_epoch,
        validation_data=val_ds,
        callbacks=[val_f1_callback, check_point, early_stop, reduce_lr, f1_callback],
    )
    return checkpoint_path
#------------------------------------------------------------------------------------------------------------------

def build_soft_vote_ensemble(model_paths):
    '''
        builds one model that averages probabilities from all branch models
    '''

    branch_models = []
    for model_index, model_path in enumerate(model_paths):
        branch_model = tf.keras.models.load_model(model_path, compile=False)
        branch_model.name = "branch_model_{}".format(model_index + 1)
        branch_model.trainable = False
        branch_models.append(branch_model)

    ensemble_input = tf.keras.Input(shape=(INPUTS_r,), name='ensemble_input')
    branch_outputs = [branch_model(ensemble_input, training=False) for branch_model in branch_models]
    ensemble_output = tf.keras.layers.Average(name='soft_vote_average')(branch_outputs)

    return tf.keras.Model(inputs=ensemble_input, outputs=ensemble_output, name='ersa_soft_vote_ensemble')
#------------------------------------------------------------------------------------------------------------------

def cleanup_temp_models(model_paths):
    for model_path in model_paths:
        if os.path.exists(model_path):
            os.remove(model_path)
#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME), compile=False)
    res = final_model.predict(test_ds)
    xres = [ tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res


    # For multiclass

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # End of Multiclass

    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        else:
            xmet =print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)
    # Ask for arguments for each metric
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    xdf_train_full = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_train, xdf_val = train_test_split(
        xdf_train_full,
        test_size=VALIDATION_SPLIT,
        random_state=SPLIT_RANDOM_STATE,
        stratify=xdf_train_full['target'],
    )

    val_ds = read_data(xdf_val, OUTPUTS_a)
    train_splits = build_submodel_train_splits(
        xdf_train=xdf_train,
        class_names=class_names,
        num_submodels=NUM_SUBMODELS,
        anchor_class='class5',
        seed=SPLIT_RANDOM_STATE,
    )

    temp_model_paths = []
    try:
        for model_index, train_split_df in enumerate(train_splits):
            tf.keras.backend.clear_session()
            train_ds = read_data_train_standard(
                dset_df=train_split_df,
                num_classes=OUTPUTS_a,
                seed=SPLIT_RANDOM_STATE + model_index,
            )
            temp_model_path = train_single_submodel(
                model_index=model_index,
                train_ds=train_ds,
                val_ds=val_ds,
            )
            temp_model_paths.append(temp_model_path)

        tf.keras.backend.clear_session()
        ensemble_model = build_soft_vote_ensemble(temp_model_paths)
        ensemble_model.save('model_{}.keras'.format(NICKNAME))
        save_model(ensemble_model)
    finally:
        cleanup_temp_models(temp_model_paths)

    # Preprocessing Test dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds= read_data(xdf_dset, OUTPUTS_a)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------

