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
from ersa_dataloader import build_standard_dataset, build_class_balanced_dataset

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
F1_EVAL_MAX_BATCHES = 100
VALIDATION_SPLIT = 0.15
SPLIT_RANDOM_STATE = 42
EARLY_STOP_PATIENCE = 20

## Image processing
CHANNELS = 3
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

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

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

def read_data_balanced(dset_df, num_classes):
    '''
          reads the train dataset with class-balanced sampling
    '''

    ds_inputs = np.array(DATA_DIR + dset_df['id'])
    ds_targets = get_target(dset_df, num_classes)

    return build_class_balanced_dataset(
        ds_inputs=ds_inputs,
        ds_targets=ds_targets,
        process_path_fn=process_path_train,
        batch_size=BATCH_SIZE,
        autotune=AUTOTUNE,
        epoch_samples=len(ds_inputs),
    )
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

    save_model(model) #print Summary
    return model
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds):
    '''
        train the model
    '''

    check_point = tf.keras.callbacks.ModelCheckpoint(
        'model_{}.keras'.format(NICKNAME),
        monitor='val_f1_macro',
        mode='max',
        save_best_only=True,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_macro',
        mode='max',
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=False,
        min_delta=1e-4,
    )
    val_f1_callback = ValidationMacroF1Callback(val_ds)
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
        callbacks=[val_f1_callback, check_point, early_stop, f1_callback],
    )
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

    train_ds = read_data_balanced(xdf_train, OUTPUTS_a)
    val_ds = read_data(xdf_val, OUTPUTS_a)
    train_func(train_ds, val_ds)

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

