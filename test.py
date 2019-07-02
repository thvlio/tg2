# import necessary packages
import tensorflow as tf
import matplotlib
import argparse
import pandas as pd
import numpy as np
import time
import math
import os
import keras.backend.tensorflow_backend as K
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import L1L2
from keras.models import model_from_json
from keras.utils import plot_model

# import files
import config

def main():
    ''' Main function. '''

    # setup argument parser
    ap = argparse.ArgumentParser('Training parameters')
    ap.add_argument('-d', '--dataset', type=str,
        help='Name of dataset to use')
    ap.add_argument('-b', '--batch-size', type=int,
        help='Number of samples per batch')
    ap.add_argument('-l', '--layer-units', type=int, nargs='+',
        help='List of number of units per layer')
    ap.add_argument('-c', '--cpu', action='store_true',
        help='If present, do not attempt configure GPU usage')
    ap.add_argument('-i', '--iter', type=int,
        help='Only execute a certain training iteration')
    ap.add_argument('-p', '--plot', action='store_true',
        help='If present, plot results')
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
    input_arch = (0, args['batch_size'], args['layer_units'])
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
    test_x = test_set.iloc[:, :-3].values.astype('float32')
    test_y = test_set.iloc[:, -3:-1].values.astype('float32')

    # input and output size
    input_size = train_x.shape[1]
    output_size = train_y.shape[1]

    # loading scalers
    print('Loading scalers')
    scaler_x = joblib.load(os.path.join(data_folder, 'scaler_x.pkl'))
    scaler_y = joblib.load(os.path.join(data_folder, 'scaler_y.pkl'))

    # normalize the features
    print('Normalizing features')
    train_x = scaler_x.transform(train_x)
    test_x = scaler_x.transform(test_x)

    # normalize the output
    print('Normalizing output')
    train_y = scaler_y.transform(train_y)
    test_y = scaler_y.transform(test_y)

    # reshape input to be 3D [samples, timesteps, features]
    print('Reshaping data')
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    # dataframes containing all loss curves
    history_dfs = []

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

        # loop on the number of iterations
        for it in range(num_iter):

            # adjust iteration number
            if args['iter'] is not None:
                it = args['iter']

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

            # load model
            print('Loading serialized model')
            model_json = None
            with open(os.path.join(model_folder, 'model.json'), 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)

            # load best model weights
            print('Loading best model weights')
            best_model_path = os.path.join(model_folder, 'best-model.hdf5')
            model.load_weights(best_model_path)

            # compilation
            print('Compiling model')
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mse']
            )

            plot_model(model=model, to_file=os.path.join(model_folder, 'model.png'),
                show_shapes=True, show_layer_names=True)

            # evaluate with keras
            print('Evaluating with Keras (normalized)')
            score, mse = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
            rmse = math.sqrt(mse)
            print('Score: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}'.format(score, mse, rmse))

            # save rmse
            rmses.append(rmse)

            # write rmse on disk
            with open(os.path.join(model_folder, 'rmse.txt'), 'w') as res_file:
                res_file.write('{}\n'.format(rmse))
            
            # save the loss curves
            history_dfs.append((model_string, pd.read_csv(filepath_or_buffer=os.path.join(model_folder, 'history.csv'))))

            # plot the results
            if args['plot']:

                # get the whole dataset
                print('Getting dataset')
                dataset = pd.concat((train_set, test_set, val_set))
                dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
                dataset = dataset.sort_values(['datetime'])
                dataset = dataset.reset_index(drop=True)
                complete_x = dataset.iloc[:, :-3].values.astype('float32')
                complete_y = dataset.iloc[:, -3:-1].values.astype('float32')
                complete_x = scaler_x.transform(complete_x)
                complete_y = scaler_y.transform(complete_y)
                complete_x = complete_x.reshape((complete_x.shape[0], 1, complete_x.shape[1]))

                # dataset with estimates
                print('Running estimates for the dataset')
                dataset_yhat = model.predict(complete_x)
                inv_yhat = scaler_y.inverse_transform(dataset_yhat)
                inv_y = scaler_y.inverse_transform(complete_y)

                # append data to the dataset dataframe
                predictions = pd.DataFrame(data=inv_yhat, columns=['jx_pred', 'jy_pred'])
                dataset_ext = pd.concat((dataset, predictions), axis=1)

                # plot the results
                print('Plotting predictions')
                pretty_name = "Oito" if dataset_name == "oito" else "Prédio" if dataset_name == "predio" else "Other"
                dataset_ext['secs'] = (dataset_ext['datetime']-dataset_ext['datetime'][0]).dt.total_seconds()
                titles_columns = [('Eixo Y do controle (eixo X do robô)', ['jy_gt', 'jy_pred'], 'Velocidade linear (m/s)'),
                    ('Eixo X do controle (eixo Z do robô)', ['jx_gt', 'jx_pred'], 'Velocidade angular (rad/s)')]
                fig = pyplot.figure(figsize=config.figsizej)
                axis = fig.subplots(nrows=len(titles_columns), ncols=1, sharex=True)
                for i, (title, columns, ylabel) in enumerate(titles_columns):
                    axis[i].plot(dataset_ext['secs'], dataset_ext.loc[:, columns])
                    axis[i].set_title(title)
                    if i+1 == len(titles_columns):
                        axis[i].set_xlabel('Tempo (s)')
                    axis[i].set_ylabel('Deslocamento da alavanca')
                    axis[i].legend(columns)
                    axis[i].grid(True, which='both')
                    par = axis[i].twinx()
                    par.spines["right"].set_position(("axes", 1.0))
                    par.set_frame_on(True)
                    par.patch.set_visible(False)
                    for sp in par.spines.values():
                        sp.set_visible(False)
                    par.spines["right"].set_visible(True)
                    par.yaxis.set_label_position('right')
                    par.yaxis.set_ticks_position('right')
                    bottom, top = axis[i].get_ylim()
                    par.set_ylim(bottom/2, top/2)
                    par.set_ylabel(ylabel)
                pyplot.suptitle(f'Comandos do usuário | {pretty_name} | ({", ".join([str(units) for units in layer_units])})-{batch_size}-{it}')
                pyplot.savefig(os.path.join(model_folder, 'predictions.png'), bbox_inches='tight')

                # plot history
                print('Plotting history')
                fig = pyplot.figure(figsize=config.figsizel)
                history = pd.read_csv(os.path.join(model_folder, 'history.csv'))
                pyplot.plot(history['epoch'], history['loss'], label='train loss')
                pyplot.plot(history['epoch'], history['val_loss'], label='val loss')
                pyplot.title(f'Perda x época | {pretty_name} | ({", ".join([str(units) for units in layer_units])})-{batch_size}-{it}')
                pyplot.xlabel('Época')
                pyplot.ylabel('Perda')
                pyplot.grid(True, which='both')
                pyplot.legend()
                pyplot.savefig(os.path.join(model_folder, 'history.png'), bbox_inches='tight')

                # close the figures
                pyplot.close('all')
            
            # cleanup
            K.clear_session()
            del model

        # get the rmse mean and standard deviation
        print('Evaluating mean and stardard deviation')
        rmse_mean = np.mean(rmses)
        rmse_means.append(rmse_mean)
        rmse_std = np.std(rmses)
        rmse_stds.append(rmse_std)
        print('RMSE: {:.4f} \u00B1 {:.4f}'.format(rmse_mean, rmse_std))
    
    # print the total duration
    print('Total testing time: {}'.format(time.time()-tstart))

    # plot all results
    if args['plot']:
        pyplot.show()

    # plot all loss curves
    if args['plot']:
        print('Plotting all loss curves')
        pyplot.figure()
        for model, df in history_dfs:
            pyplot.title('Training and validation loss curves')
            pyplot.plot(df['epoch'], df['loss'], label=model)
            pyplot.plot(df['epoch'], df['val_loss'], label=model)
            pyplot.legend()
        pyplot.show()

# if it is the main file
if __name__ == '__main__':
    main()
