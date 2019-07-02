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

    # best models
    best_rmses = np.array([1.0] * 10)
    best_archs = np.array([None] * 10)

    # loop on different architectures
    for epochs, batch_size, layer_units in config.network_archs:

        # rmse list to get rmse mean and standard deviation
        rmses = []

        # print the architecture and some other info
        #print('Architecture & info:')
        #print('\tUnits per layer: {}'.format(layer_units))
        #print('\tBatch size: {}'.format(batch_size))

        # loop on the number of iterations
        for it in range(config.num_iter):

            # print the iteration
            #print('\tIteration {}'.format(it))

            # model string and folder
            model_string = '{}'.format(
                '_'.join([
                    'L' + '-'.join([str(units) for units in layer_units]),
                    'B' + str(batch_size),
                    'I' + str(it)
                ])
            )
            model_folder = os.path.join('models', config.dataset_name, model_string)

            # read results from file
            #print('\tReading RMSE from file')
            with open(os.path.join(model_folder, 'rmse.txt'), 'r') as res_file:
                rmse = float(res_file.read())

            # save rmse
            rmses.append(rmse)

            # check if its in the top 10 models so far
            if rmse < np.max(best_rmses):
                idx = np.searchsorted(best_rmses, rmse)
                best_rmses = np.insert(best_rmses, idx, rmse)[:-1]
                best_archs = np.insert(best_archs, idx, model_string)[:-1]

        # get the rmse mean and standard deviation
        #print('\tEvaluating mean and stardard deviation')
        rmse_mean = np.mean(rmses)
        rmse_means.append(rmse_mean)
        rmse_std = np.std(rmses)
        rmse_stds.append(rmse_std)
        #print('\tRMSE: {:.4f} \u00B1 {:.4f}'.format(rmse_mean, rmse_std))

    # print all RMSEs in descending order
    print('\nSorting models by RMSE:')
    idxs = np.arange(0, len(rmse_means)) #idxs = np.argsort(rmse_means)[::-1]
    rmse_means = np.array(rmse_means)[idxs]
    rmse_stds = np.array(rmse_stds)[idxs]
    #archs = config.network_archs[idxs]
    for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
        epochs, batch_size, layer_units = config.network_archs[idxs[i]]
        print('\tArchitecture & info:')
        print('\t\tUnits per layer: {}'.format(layer_units))
        print('\t\tBatch size: {}'.format(batch_size))
        print('\t\tRMSE: {mean:.4f} \u00B1 {std:.4f}'.format(mean=mean, std=std))

    # best models
    best_rmses = best_rmses[::-1]; best_archs = best_archs[::-1]; print()
    for i, (model_string, rmse) in enumerate(zip(best_archs, best_rmses)):
        print('Number {} model: {}, with RMSE: {:.4f}'.format(10-i, model_string, rmse))

# if it is the main file
if __name__ == '__main__':
    main()
