# import necessary packages
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot, dates
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection

# import files
import config

def main():
    ''' Main function. '''

    # get the whole dataset
    print('Reading dataset')
    data_folder = os.path.join('data', config.dataset_name)
    train_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'train', 'data.csv'), parse_dates=['datetime'])
    test_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'test', 'data.csv'), parse_dates=['datetime'])
    val_set = pd.read_csv(filepath_or_buffer=os.path.join(data_folder, 'val', 'data.csv'), parse_dates=['datetime'])
    dataset = pd.concat((train_set, test_set, val_set)).sort_values(['datetime']).reset_index(drop=True)

    # plot the correlation between all variables
    # print('Plotting the correlation between variables')
    # rdataset = dataset.drop('datetime', axis=1)
    # fig, ax = pyplot.subplots(figsize=(10, 7))
    # fig.colorbar(ax.matshow(rdataset.corr(), cmap='coolwarm', vmin=-1, vmax=1))
    # pyplot.xticks(np.arange(len(rdataset.columns)), rdataset.columns, rotation=30)
    # pyplot.yticks(np.arange(len(rdataset.columns)), rdataset.columns)
    # pyplot.savefig(os.path.join(data_folder, 'corr-mat.png'), bbox_inches='tight')
    # pd.plotting.scatter_matrix(rdataset, alpha=0.3, figsize=(16, 9))
    # pyplot.savefig(os.path.join(data_folder, 'corr-graph.png'), bbox_inches='tight')

    # plot data
    print('Plotting data')
    titles_columns = [('Eixo Y do controle (eixo X do robô)', ['jy'], 'Velocidade linear (m/s)'),
        ('Eixo X do controle (eixo Z do robô)', ['jx'], 'Velocidade angular (rad/s)')]
    fig = pyplot.figure(figsize=config.figsizes)
    dataset['secs'] = (dataset['datetime']-dataset['datetime'][0]).dt.total_seconds()
    # dataset['xr'] = dataset['x'] - dataset['x'][0]; dataset['yr'] = dataset['y'] - dataset['y'][0]
    if config.dataset_name == '1205a':
        train1 = train_set.iloc[:4791, :].copy(); train2 = train_set.iloc[4791:, :].copy()
        plist = [('train1', train1, 'C0'), ('train2', train2, 'C0'), ('test', test_set, 'C1'), ('val', val_set, 'C2')]
        train1['secs'] = (train1['datetime']-dataset['datetime'][0]).dt.total_seconds()
        train2['secs'] = (train2['datetime']-dataset['datetime'][0]).dt.total_seconds()
        test_set['secs'] = (test_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
        val_set['secs'] = (val_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
    elif config.dataset_name == '1205c':
        plist = [('train', train_set, 'C0'), ('test', test_set, 'C1'), ('val', val_set, 'C2')]
        train_set['secs'] = (train_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
        test_set['secs'] = (test_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
        val_set['secs'] = (val_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
    axisp = pyplot.subplot(1, 2, 2)
    for (name, dset, clr) in plist:
        axisp.plot(dset['x'], dset['y'], c=clr, label='train' if name=='train1' else '' if name=='train2' else f'{name}')
    # axisp.plot(dataset['xr'], dataset['yr'])
    axisp.set_title('Caminho percorrido')
    axisp.set_xlabel('Posição no eixo X (m)')
    axisp.set_ylabel('Posição no eixo Y (m)')
    axisp.grid(True, which='both')
    for i, (title, columns, ylabel) in enumerate(titles_columns):
        axisi = pyplot.subplot(len(titles_columns), 2, (i)*2+1)
        for (name, dset, clr) in plist:
            for c in columns:
                axisi.plot(dset['secs'], dset[c], c=clr, label=f'{c}_train' if name=='train1' else '' if name=='train2' else f'{c}_{name}')
        axisi.set_title(title)
        if i+1 == len(titles_columns):
            axisi.set_xlabel('Tempo (s)')
        axisi.set_ylabel('Deslocamento da alavanca')
        axisi.legend()
        axisi.grid(True, which='both')
        # axisi.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        par = axisi.twinx()
        par.spines["right"].set_position(("axes", 1.0))
        par.set_frame_on(True)
        par.patch.set_visible(False)
        for sp in par.spines.values():
            sp.set_visible(False)
        par.spines["right"].set_visible(True)
        par.yaxis.set_label_position('right')
        par.yaxis.set_ticks_position('right')
        bottom, top = axisi.get_ylim()
        par.set_ylim(bottom/2, top/2)
        par.set_ylabel(ylabel)
    pyplot.savefig(os.path.join(data_folder, 'joined.png'), bbox_inches='tight')

    '''
    # plot data
    print('Plotting data')
    titles_columns = [('Eixo Y do controle (eixo X do robô)', ['jy'], 'Velocidade linear (m/s)'),
        ('Eixo X do controle (eixo Z do robô)', ['jx'], 'Velocidade angular (rad/s)')]
    fig = pyplot.figure(figsize=config.figsizes)

    dataset['xr'] = dataset['x'] - dataset['x'][0]
    dataset['yr'] = dataset['y'] - dataset['y'][0]
    dataset['secs'] = (dataset['datetime']-dataset['datetime'][0]).dt.total_seconds()
    axisp = pyplot.subplot(1, 2, 2)
    points = np.array([dataset['xr'].values, dataset['yr'].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = pyplot.Normalize(dataset['secs'].min(), dataset['secs'].max())
    lc = LineCollection(segments, cmap='winter', norm=norm)
    lc.set_array(dataset['secs'].values); lc.set_linewidth(2)
    line = axisp.add_collection(lc)
    fig.colorbar(line, ax=axisp)
    pyplot.autoscale()
    # axisp.plot(dataset['xr'], dataset['yr'])
    axisp.set_title('Caminho percorrido')
    axisp.set_xlabel('Posição no eixo X (m)')
    axisp.set_ylabel('Posição no eixo Y (m)')
    axisp.grid(True, which='both')

    if config.dataset_name == '1205a':
        train1 = train_set.iloc[:4791, :]; train2 = train_set.iloc[4791:, :]
        plist = [('train1', train1, 'winter'), ('train2', train2, 'winter'), ('test', test_set, 'autumn'), ('val', val_set, 'summer')]
        train1['secs'] = (train1['datetime']-dataset['datetime'][0]).dt.total_seconds()
        train2['secs'] = (train2['datetime']-dataset['datetime'][0]).dt.total_seconds()
        test_set['secs'] = (test_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
        val_set['secs'] = (val_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
    elif config.dataset_name == '1205c':
        plist = [('train', train_set, 'winter'), ('test', test_set, 'autumn'), ('val', val_set, 'summer')]
        train_set['secs'] = (train_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
        test_set['secs'] = (test_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
        val_set['secs'] = (val_set['datetime']-dataset['datetime'][0]).dt.total_seconds()
    for i, (title, columns, ylabel) in enumerate(titles_columns):
        axisi = pyplot.subplot(len(titles_columns), 2, (i)*2+1)
        for (name, dset, clr) in plist:
            for c in columns:

                points = np.array([dset['secs'].values, dset[c].values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=clr, norm=norm)
                lc.set_array(dset['secs'].values); lc.set_linewidth(2)
                line = axisi.add_collection(lc)
                line.set_label(f'{c}_train' if name=='train1' else '' if name=='train2' else f'{c}_{name}')
                pyplot.autoscale()

                # axisi.plot(dset['datetime'], dset[c], c=clr, label=f'{c}_train' if name=='train1' else '' if name=='train2' else f'{c}_{name}')
        
        axisi.set_title(title)
        if i+1 == len(titles_columns):
            axisi.set_xlabel('Tempo (s)')
        axisi.set_ylabel('Deslocamento da alavanca')
        axisi.legend()
        axisi.grid(True, which='both')
        # axisi.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        par = axisi.twinx()
        par.spines["right"].set_position(("axes", 1.0))
        par.set_frame_on(True)
        par.patch.set_visible(False)
        for sp in par.spines.values():
            sp.set_visible(False)
        par.spines["right"].set_visible(True)
        par.yaxis.set_label_position('right')
        par.yaxis.set_ticks_position('right')
        bottom, top = axisi.get_ylim()
        par.set_ylim(bottom/2, top/2)
        par.set_ylabel(ylabel)
    pyplot.savefig(os.path.join(data_folder, 'joined.png'), bbox_inches='tight')
    '''

    '''
    # plot data
    print('Plotting data')
    titles_columns = [('Eixo Y do controle (eixo X do robô)', ['jy']),
        ('Eixo X do controle (eixo Z do robô)', ['jx'])]
    fig = pyplot.figure(figsize=config.figsizej)
    axis = fig.subplots(nrows=len(titles_columns), ncols=1, sharex=True)
    for i, (title, columns) in enumerate(titles_columns):
        axis[i].plot(dataset['datetime'], dataset.loc[:, columns])
        axis[i].set_title(title)
        if i+1 == len(titles_columns):
            axis[i].set_xlabel('Tempo')
        axis[i].set_ylabel('Valor')
        axis[i].legend(columns)
        axis[i].grid(True, which='both')
    pyplot.savefig(os.path.join(data_folder, 'data.png'), bbox_inches='tight')

    # save the image plots
    fig = pyplot.figure(figsize=config.figsizep)
    dataset['xr'] = dataset['x'] - dataset['x'][0]
    dataset['yr'] = dataset['y'] - dataset['y'][0]
    pyplot.plot(dataset['xr'], dataset['yr'])
    pyplot.title('Caminho percorrido')
    pyplot.xlabel('Posição no eixo X (m)')
    pyplot.ylabel('Posição no eixo Y (m)')
    pyplot.grid(True, which='both')
    pyplot.savefig(os.path.join(data_folder, 'paths.png'), bbox_inches='tight')

    # plot data
    print('Plotting data')
    titles_columns = [('Eixo Y do controle (eixo X do robô)', ['jy'], 'Velocidade linear (m/s)'),
        ('Eixo X do controle (eixo Z do robô)', ['jx'], 'Velocidade angular (rad/s)')]
    fig = pyplot.figure(figsize=config.figsizej)
    axis = fig.subplots(nrows=len(titles_columns), ncols=1, sharex=True)
    if config.dataset_name == '1205a':
        train1 = train_set.iloc[:4791, :]; train2 = train_set.iloc[4791:, :]
        plist = [('train1', train1, 'C0'), ('train2', train2, 'C0'), ('test', test_set, 'C1'), ('val', val_set, 'C2')]
    elif config.dataset_name == '1205c':
        plist = [('train', train_set, 'C0'), ('test', test_set, 'C1'), ('val', val_set, 'C2')]
    for i, (title, columns, ylabel) in enumerate(titles_columns):
        for (name, dset, clr) in plist:
            for c in columns:
                axis[i].plot(dset['datetime'], dset[c], c=clr, label=f'{c}_train' if name=='train1' else '' if name=='train2' else f'{c}_{name}')
        axis[i].set_title(title)
        if i+1 == len(titles_columns):
            axis[i].set_xlabel('Tempo (HH:MM)')
        axis[i].set_ylabel('Deslocamento da alavanca')
        axis[i].legend()
        axis[i].grid(True, which='both')
        axis[i].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        par = axis[i].twinx()
        par.spines["left"].set_position(("axes", -0.1))
        par.set_frame_on(True)
        par.patch.set_visible(False)
        for sp in par.spines.values():
            sp.set_visible(False)
        par.spines["left"].set_visible(True)
        par.yaxis.set_label_position('left')
        par.yaxis.set_ticks_position('left')
        bottom, top = axis[i].get_ylim()
        par.set_ylim(bottom/2, top/2)
        par.set_ylabel(ylabel)
    pyplot.savefig(os.path.join(data_folder, 'datacoded.png'), bbox_inches='tight')
    '''

# if it is the main file
if __name__ == '__main__':
    main()
