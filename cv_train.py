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
from sklearn.model_selection import KFold
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
    ap.add_argument('-f', '--fold', type=int,
        help='Only use a certain fold')
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
    num_iter = 1 if args['fold'] is not None else config.num_iter

    # load the data as pandas dataframes
    print('Loading data as dataframes')
    data_folder = os.path.join('data', dataset_name)
    train_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'train', 'data.csv'))
    val_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'val', 'data.csv'))
    test_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'test', 'data.csv'))

    # get the whole dataset
    print('Getting dataset')
    dataset = pd.concat((train_set, test_set, val_set))
    dataset = dataset.reset_index(drop=True)

    # load data as matrices
    print('Loading data as matrices')
    complete_x = dataset.iloc[:, :-3].values.astype('float32')
    complete_y = dataset.iloc[:, -3:-1].values.astype('float32')

    # input and output size
    input_size = (1, complete_x.shape[1])
    output_size = complete_y.shape[1]

    # normalize the features
    print('Normalizing features')
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x.fit(complete_x)
    complete_x = scaler_x.transform(complete_x)

    # normalize the output
    print('Normalizing output')
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(complete_y)
    complete_y = scaler_y.transform(complete_y)

    # reshape input to be 3D [samples, timesteps, features]
    print('Reshaping data')
    complete_x = complete_x.reshape((complete_x.shape[0], 1, complete_x.shape[1]))

    # create the folds for k-folds cross-validation
    kfold = KFold(n_splits=num_iter, shuffle=True, random_state=1337)

    # rmse info
    rmse_means = []
    rmse_stds = []

    # timing
    tstart = time.time()

    # loop on different architectures
    for epochs, batch_size, layer_units in network_archs:

        # rmse list to get rmse mean and standard deviation
        rmses = []

        # print the architecture and some other info
        print('Architecture & info:')
        print('\tUnits per layer: {}'.format(layer_units))
        print('\tBatch size: {}'.format(batch_size))

        '''
        # new figure for the history plots
        fig = pyplot.figure(figsize=config.figsizej)
        '''

        # loop on the number of folds
        for fold, (train, test) in enumerate(kfold.split(complete_x, complete_y)):

            # adjust fold number
            if args['fold'] is not None:
                fold = args['fold']

            # timing
            t0 = time.time()

            # print the fold
            print('Fold {}'.format(fold))

            # model string and folder
            model_string = '{}'.format(
                '_'.join([
                    'L' + '-'.join([str(units) for units in layer_units]),
                    'B' + str(batch_size)
                ])
            )
            model_folder = os.path.join('models', 'cross-validation', dataset_name, model_string)

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
                            input_shape=input_size,
                            return_sequences=True))
                    else:
                        model.add(LSTM(units=num_units,
                            dropout=0.2, recurrent_dropout=0.2,
                            input_shape=input_size))
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
                filepath=os.path.join(model_folder, f'best-model-{fold}.hdf5'),
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
                filename=os.path.join(model_folder, f'history-{fold}.csv')
            )
            
            # fit network
            print('Training')
            history = model.fit(complete_x[train], complete_y[train],
                batch_size=batch_size, epochs=epochs,
                validation_data=(complete_x[test], complete_y[test]),
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
            '''

            # evaluate with keras
            print('Evaluating with Keras (normalized)')
            score, mse = model.evaluate(complete_x[test], complete_y[test], batch_size=batch_size, verbose=1)
            rmse = math.sqrt(mse)
            print('Score: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}'.format(score, mse, rmse))

            # save rmse
            rmses.append(rmse)

            # write rmse on disk
            with open(os.path.join(model_folder, f'rmse-{fold}.txt'), 'w') as res_file:
                res_file.write('{}\n'.format(rmse))

            # cleanup
            K.clear_session()
            del model

            # timing
            t1 = time.time()

            # print the duration
            print('Training time: {}'.format(t1-t0))

        '''
        # save the history plots
        pyplot.savefig(os.path.join(model_folder, 'history.png'))
        '''

        # get the rmse mean and standard deviation
        print('Evaluating mean and stardard deviation')
        rmse_mean = np.mean(rmses)
        rmse_means.append(rmse_mean)
        rmse_std = np.std(rmses)
        rmse_stds.append(rmse_std)
        print('RMSE: {:.4f} \u00B1 {:.4f}'.format(rmse_mean, rmse_std))

        # write rmse info on disk
        with open(os.path.join(model_folder, 'rmse-info.txt'), 'w') as res_file:
            res_file.write('{} {}\n'.format(rmse_mean, rmse_std))

    # print the total duration
    print('Total training time: {}'.format(time.time()-tstart))

    # print all RMSEs in ascending order
    print('\nSorting models by RMSE:')
    idxs = np.argsort(rmse_means)
    rmse_means = np.array(rmse_means)[idxs]
    rmse_stds = np.array(rmse_stds)[idxs]
    #archs = config.network_archs[idxs]
    for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
        epochs, batch_size, layer_units = config.network_archs[idxs[i]]
        print('\tArchitecture & info:')
        print('\t\tUnits per layer: {}'.format(layer_units))
        print('\t\tBatch size: {}'.format(batch_size))
        print('\t\tRMSE: {mean:.4f} \u00B1 {std:.4f}'.format(mean=mean, std=std))

# if it is the main file
if __name__ == '__main__':
    main()
