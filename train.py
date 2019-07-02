# import necessary packages
import tensorflow as tf
import argparse
import pandas as pd
import numpy as np
import keras
import math
import time
import os
import keras.backend.tensorflow_backend as K
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.models import model_from_json

# import files
import config

def main():
    ''' Main function. '''

    # setup argument parser
    ap = argparse.ArgumentParser('Training parameters')
    ap.add_argument('-d', '--dataset', type=str,
        help='Name of dataset to use')
    ap.add_argument('-e', '--epochs', type=int,
        help='Number of epochs to train')
    ap.add_argument('-b', '--batch-size', type=int,
        help='Number of samples per batch')
    ap.add_argument('-l', '--layer-units', type=int, nargs='+',
        help='List of number of units per layer')
    ap.add_argument('-c', '--cpu', action='store_true',
        help='If present, do not attempt configure GPU usage')
    ap.add_argument('-i', '--iter', type=int,
        help='Only execute a certain training iteration')
    args = vars(ap.parse_args())

    # configure tf session
    if not args['cpu']:
        print('Configuring TF session')
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        session = tf.Session(config=config_tf)
        K.set_session(session)

    # save parameters and overwrite config if all parameters were given
    print('Parsing input parameters')
    dataset_name = args['dataset'] if args['dataset'] is not None else config.dataset_name
    input_arch = (args['epochs'], args['batch_size'], args['layer_units'])
    network_archs = [input_arch] if None not in input_arch else config.network_archs
    num_iter = 1 if args['iter'] is not None else config.num_iter

    # load the data as pandas dataframes
    print('Loading data as dataframes')
    data_folder = os.path.join('data', dataset_name)
    train_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'train', 'data.csv'))
    val_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'val', 'data.csv'))
    test_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'test', 'data.csv'))

    # load data as matrices
    print('Loading data as matrices')
    train_x = train_set.iloc[:, :-3].values.astype('float32')
    train_y = train_set.iloc[:, -3:-1].values.astype('float32')
    val_x = val_set.iloc[:, :-3].values.astype('float32')
    val_y = val_set.iloc[:, -3:-1].values.astype('float32')

    # input and output size
    input_size = train_x.shape[1]
    output_size = train_y.shape[1]

    # normalize the features
    print('Normalizing features')
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x.fit(train_x)
    train_x = scaler_x.transform(train_x)
    val_x = scaler_x.transform(val_x)

    # normalize the output
    print('Normalizing output')
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(train_y)
    train_y = scaler_y.transform(train_y)
    val_y = scaler_y.transform(val_y)

    # serialize scalers
    print('Serializing scalers')
    joblib.dump(scaler_x, os.path.join(data_folder, 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(data_folder, 'scaler_y.pkl'))

    # reshape input to be 3D [samples, timesteps, features]
    print('Reshaping data')
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    val_x = val_x.reshape((val_x.shape[0], 1, val_x.shape[1]))

    # timing
    tstart = time.time()

    # loop on different architectures
    for epochs, batch_size, layer_units in network_archs:

        # print the architecture and some other info
        print('Architecture & info:')
        print('\tUnits per layer: {}'.format(layer_units))
        print('\tBatch size: {}'.format(batch_size))

        # loop on the number of iterations
        for it in range(num_iter):

            # adjust iteration number
            if args['iter'] is not None:
                it = args['iter']

            # timing
            t0 = time.time()

            # print the iteration
            print('\tIteration {}'.format(it))

            # model string and folder
            model_string = '{}'.format(
                '_'.join([
                    'L' + '-'.join([str(units) for units in layer_units]),
                    'B' + str(batch_size),
                    'I' + str(it)
                ])
            )
            model_folder = os.path.join('models', dataset_name, model_string)

            # create the necessary directories
            if not os.path.isdir(model_folder):
                os.makedirs(model_folder)

            # design the network
            print('Creating model')
            model = Sequential()
            for i, num_units in enumerate(layer_units):
                if i == 0:
                    if i+1 < len(layer_units):
                        model.add(LSTM(units=num_units,
                            dropout=0.2, recurrent_dropout=0.2,
                            input_shape=(train_x.shape[1], train_x.shape[2]),
                            return_sequences=True))
                    else:
                        model.add(LSTM(units=num_units,
                            dropout=0.2, recurrent_dropout=0.2,
                            input_shape=(train_x.shape[1], train_x.shape[2])))
                else:
                    if i+1 < len(layer_units):
                        model.add(LSTM(units=num_units,
                            dropout=0.2, recurrent_dropout=0.2,
                            return_sequences=True))
                    else:
                        model.add(LSTM(units=num_units,
                            dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(output_size))

            # serialize model to JSON
            print('Serializing model')
            model_json = model.to_json()
            with open(os.path.join(model_folder, 'model.json'), 'w') as json_file:
                json_file.write(model_json)

            # save model diagram
            '''
            keras.utils.plot_model(model=model, to_file=os.path.join(model_folder, 'model.png'),
                show_shapes=True, show_layer_names=True, rankdir='LR')
            '''

            # compilation
            print('Compiling model')
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mse']
            )

            # setup callbacks
            print('Configuring callbacks')
            best_model = ModelCheckpoint(
                filepath=os.path.join(model_folder, 'best-model.hdf5'),
                verbose=0,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
            early_stopping = EarlyStopping(
                monitor='val_loss',
                verbose=0,
                patience=math.log2(batch_size)*20-50
            )
            '''
            tensorboard = TensorBoard(
                log_dir=model_folder,
                batch_size=batch_size,
                write_graph=False
            )
            '''
            csv_logger = CSVLogger(
                filename=os.path.join(model_folder, 'history.csv')
            )
            
            # fit network
            print('Training')
            history = model.fit(train_x, train_y,
                batch_size=batch_size, epochs=epochs,
                validation_data=(val_x, val_y),
                callbacks=[best_model, early_stopping, csv_logger],
                verbose=1,
                shuffle=True
            )

            # plot history
            '''
            print('Plotting history')
            pyplot.plot(history.epoch, history.history['loss'], label='train loss')
            pyplot.plot(history.epoch, history.history['val_loss'], label='val loss')
            pyplot.title('loss x epoch')
            pyplot.xlabel('epoch')
            pyplot.ylabel('loss')
            pyplot.legend()
            pyplot.show()
            '''

            # cleanup
            K.clear_session()
            del model

            # timing
            t1 = time.time()

            # print the duration
            print('Training time: {}'.format(t1-t0))

    # print the total duration
    print('Total training time: {}'.format(time.time()-tstart))

# if it is the main file
if __name__ == '__main__':
    main()
