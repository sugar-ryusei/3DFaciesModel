import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import time
import os
import tensorflow as tf

import dataset
import resunet

def train(
    model_name,
    epochs=100,
    batch_size=8,
    loss='CE',
    opt='sgd',
    augmentation=True,
    input_shape=(224, 224, 5),
    train_path='',
    test_path='',
    save_direc='',
    callback_period=50,
    lr=0.001,
    decay=1e-5
    ):

    print('prepare dataset')
    gen_train = dataset.DatasetTF(
                direc=train_path,
                batch_size=batch_size,
                augmentation=augmentation)
    gen_train.set_classes()

    ds_train = gen_train.load_dataset(shuffle=True)

    gen_test = dataset.DatasetTF(
                direc=test_path,
                batch_size=batch_size,
                augmentation=False)
    gen_test.palette = gen_train.palette
    gen_test.classes = gen_train.classes

    ds_test = gen_test.load_dataset(shuffle=False)

    print('building model')
    resunet = resunet.ResUNet(input_shape=input_shape, classes=gen_train.classes)
    model = resunet.ResUNet()

    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    if loss == 'CE':
        loss_func = [tf.keras.losses.CategoricalCrossentropy()]
    
    if opt == 'sgd':
        opt = tf.keras.optimizers.SGD(
            lr=lr,
            decay=decay)

    if save_direc != '':
        save_direc = save_direc + os.sep
    else:
        save_direc = './'
    os.makedirs(save_direc, exist_ok=True)
    log_filepath = os.path.join(save_direc + 'logs_' + model_name)
    model_out = save_direc + model_name + '.h5'
    model_out_e = save_direc + model_name + '_e{epoch:04d}.h5'

    #callback
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_filepath, histogram_freq=1, write_graph=True, write_images=True)
    callback_period = int(np.ceil(gen_train.data_length / batch_size) * callback_period)
    me_cb = tf.keras.callbacks.ModelCheckpoint(
        model_out_e, monitor='val_loss', verpose=0, save_best_only=False,
        save_wight_only=False, save_freq=callback_period)
    
    callbacks = [tb_cb, me_cb]

    model.compile(loss=loss_func, optimizer=opt, metrics=metrics)
    model.summary()

    print('training start')

    history = model.fit(
        x=ds_train,
        epochs=epochs,
        verbose=1,
        validation_data=ds_test,
        shuffle=True,
        callbacks=callbacks,
        use_multiprocessing=True
        )

    df_his = pd.DataFrame(history.history)
    df_his.to_csv(model_out.replace('.h5', '.csv'))

    model.save(model_out)

def main():
    train(model_name='resunet',
    epochs=700,
    batch_size=8,
    loss='CE',
    opt='sgd',
    augmentation=True,
    input_shape=[224, 224, 4],
    train_path='training_data/train',
    test_path='training_data/test',
    save_direc='training_result',
    callback_period=50,
    lr=0.18,
    decay=1e-5
    )

if __name__ == '__main__':
    main()