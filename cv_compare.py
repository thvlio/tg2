# import necessary packages
import numpy as np
import os

# import files
import config

def main():
    ''' Main function. '''

    # rmse info
    rmse_means = []
    rmse_stds = []

    # loop on different architectures
    for epochs, batch_size, layer_units in config.network_archs:

        # print the architecture and some other info
        #print('Architecture & info:')
        #print('\tUnits per layer: {}'.format(layer_units))
        #print('\tBatch size: {}'.format(batch_size))

        # model string and folder
        model_string = '{}'.format(
            '_'.join([
                'L' + '-'.join([str(units) for units in layer_units]),
                'B' + str(batch_size)
            ])
        )
        model_folder = os.path.join('models', 'cross-validation', config.dataset_name, model_string)

        # read info from file
        #print('\tReading RMSE from file')
        with open(os.path.join(model_folder, 'rmse-info.txt'), 'r') as res_file:
            rmse_mean, rmse_std = (res_file.read().split(' '))
            rmse_mean = float(rmse_mean); rmse_std = float(rmse_std)

        # save rmse mean and standard deviation
        #print('\tEvaluating mean and stardard deviation')
        rmse_means.append(rmse_mean)
        rmse_stds.append(rmse_std)
        #print('\tRMSE: {:.4f} \u00B1 {:.4f}'.format(rmse_mean, rmse_std))

    # print all RMSEs in descending order
    print('\nSorting models by RMSE:')
    idxs = np.argsort(rmse_means)[::-1]
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
