# import necessary packages
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection

# import files
import config

def main():
    ''' Main function. '''

    # list all folders
    print('Listing all folders')
    dirs = sorted([d for d in os.scandir('results') if d.is_dir()], key=lambda d: d.name)
    pose_files_per_dir = [sorted([f for f in os.scandir(d.path) if f.is_file() and f.name.endswith('pose.csv')], key=lambda f: f.name) for d in dirs]
    info_files_per_dir = [sorted([f for f in os.scandir(d.path) if f.is_file() and f.name.endswith('info.csv')], key=lambda f: f.name) for d in dirs]

    # read all files
    print('Reading all files')
    pose_reses_per_dir = [[pd.read_csv(filepath_or_buffer=p.path, parse_dates=['datetime']) for p in pfiles] for pfiles in pose_files_per_dir]
    info_reses_per_dir = [[pd.read_csv(filepath_or_buffer=i.path, parse_dates=['datetime']) for i in ifiles] for ifiles in info_files_per_dir]

    # to make some comparisons later
    summaries = [] # [(name, path, method, thresh, mpubt, mpredt, actt, mdiff, idvx, idvz)]

    # analyse dirs
    print('Analysing')
    for preses, ireses, pfiles, ifiles, dr in zip(pose_reses_per_dir, info_reses_per_dir, pose_files_per_dir, info_files_per_dir, dirs):

        # analyse results
        print(f'Dir: {dr.name}')
        for pres, ires, pff, iff in zip(preses, ireses, pfiles, ifiles):
            print(f'Result: {pff.name} and {iff.name}')

            # parse file name
            path = pff.name.split('_')[0]
            methodtresh = pff.name.split('_')[1].split('.')[0]
            method = ''; thresh = ''
            if methodtresh.endswith('20'):
                method = methodtresh[:-2]; thresh = methodtresh[-2:]
            elif methodtresh.endswith('5'):
                method = methodtresh[:-1]; thresh = methodtresh[-1:]
            else:
                method = methodtresh; thresh = 0

            # parse some fields of the results dataframe
            ires['ddatetime'] = pd.to_timedelta(ires['ddatetime'])
            ires['ddatetime2'] = ires['datetime'].diff().dt.total_seconds()
            ires['secs'] = (ires['datetime'] - ires['datetime'][0]).dt.total_seconds()
            pres['secs'] = (pres['datetime'] - pres['datetime'][0]).dt.total_seconds()

            # plot the results
            titles_columns = [('Eixo Y do controle (eixo X do robô)', ['jpy', 'joy'], ['winter', 'autumn'], ['C0', 'C1'], 'Velocidade linear (m/s)'),
                ('Eixo X do controle (eixo Z do robô)', ['jpx', 'jox'], ['winter', 'autumn'], ['C0', 'C1'], 'Velocidade angular (rad/s)')]
            fig = pyplot.figure(figsize=config.figsizes)
            axisp = pyplot.subplot(1, 2, 2)
            points = np.array([pres['x'].values, pres['y'].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = pyplot.Normalize(pres['secs'].min(), pres['secs'].max())
            lc = LineCollection(segments, cmap='autumn', norm=norm)
            lc.set_array(pres['secs'].values); lc.set_linewidth(2)
            line = axisp.add_collection(lc)
            fig.colorbar(line, ax=axisp)
            axisp.autoscale()
            # axisp.plot(pres['x'], pres['y'], color='C2')
            axisp.scatter(pres['x'].head(1), pres['y'].head(1), s=100, c='C4', zorder=10)
            axisp.scatter(pres['x'].tail(1), pres['y'].tail(1), s=100, c='C9', zorder=10)
            axisp.set_title('Caminho percorrido')
            axisp.set_xlabel('Posição no eixo X (m)')
            axisp.set_ylabel('Posição no eixo Y (m)')
            axisp.grid(True, which='both')
            for i, (title, columns, maps, colors, ylabel) in enumerate(titles_columns):
                axisi = pyplot.subplot(len(titles_columns), 2, (i)*2+1)
                for cm, mp, cl in zip(columns, maps, colors):
                    points = np.array([ires['secs'].values, ires[cm].values]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=mp, norm=norm)
                    lc.set_array(ires['secs'].values); lc.set_linewidth(2)
                    line = axisi.add_collection(lc)
                    line.set_label(cm)
                    axisi.autoscale()
                    # axisi.plot(ires['secs'], ires.loc[:, cm], color=cl, label=cm)
                axisi.scatter(ires['secs'].head(1), ires.loc[:, cm].head(1), s=100, c='C4', zorder=10, label='Início')
                axisi.scatter(ires['secs'].tail(1), ires.loc[:, cm].tail(1), s=100, c='C9', zorder=10, label='Fim')
                axisi.set_title(title)
                if i+1 == len(titles_columns):
                    axisi.set_xlabel('Tempo (s)')
                axisi.set_ylabel('Deslocamento da alavanca')
                axisi.legend()
                leg = axisi.get_legend()
                leg.legendHandles[0].set_color('C0')
                leg.legendHandles[1].set_color('C1')
                axisi.grid(True, which='both')
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
            pyplot.suptitle(f'Comandos e caminho percorrido | {"R" if method == "net" else "M" if method == "mean" else "D"}{thresh}')
            pyplot.savefig(os.path.join(dr.path, f'{iff.name.split(".")[0]}-graph.png'), bbox_inches='tight')

            # publishing time
            mean_pub_time = ires['ddatetime2'].mean()
            min_pub_times = ires['ddatetime2'].nsmallest()
            max_pub_times = ires['ddatetime2'].nlargest()

            # prediction time
            mean_pred_time = ires['time'].mean()
            min_pred_times = ires['time'].nsmallest()
            max_pred_times = ires['time'].nlargest()

            # get acting percentage and difference
            valid_acting = np.logical_and(ires['acting'] == 1, ires['enabled'] == 1)
            valid_instants = np.logical_and(np.logical_or(ires['moving'] == 1, ires['acting'] == 1), ires['enabled'] == 1)
            acting_perc = valid_acting.sum() / valid_instants.sum()
            mean_diff = ires.loc[valid_instants, 'diff'].mean()
            min_diffs = ires.loc[valid_instants, 'diff'].nsmallest()
            max_diffs = ires.loc[valid_instants, 'diff'].nlargest()

            # integration of the absolute differences of speed
            idvx = (ires['vx'].diff().abs() / ires['ddatetime2']).mean()
            idvz = (ires['vz'].diff().abs() / ires['ddatetime2']).mean()

            # write to summaries
            summaries.append((dr.name, path, method, thresh, mean_pub_time, mean_pred_time,
                acting_perc, mean_diff, idvx, idvz))

            # save info to file
            with open(os.path.join(dr.path, f'{iff.name.split(".")[0]}-res.txt'), 'w') as res_file:
                res_file.write(f'Mean publishing time :-> {mean_pub_time:.4f}\n')
                [res_file.write(f'\t{m:.4f}') for m in min_pub_times]; res_file.write('\n')
                [res_file.write(f'\t{m:.4f}') for m in max_pub_times]; res_file.write('\n')
                res_file.write(f'Mean prediction time :-> {mean_pred_time:.4f}\n')
                [res_file.write(f'\t{m:.4f}') for m in min_pred_times]; res_file.write('\n')
                [res_file.write(f'\t{m:.4f}') for m in max_pred_times]; res_file.write('\n')
                res_file.write(f'Acting percentage :-> {acting_perc:.4f}\n')
                res_file.write(f'Mean difference :-> {mean_diff:.4f}\n')
                [res_file.write(f'\t{m:.4f}') for m in min_diffs]; res_file.write('\n')
                [res_file.write(f'\t{m:.4f}') for m in max_diffs]; res_file.write('\n')
                res_file.write(f'Operation: ∫|dvx/dt|dt :-> {idvx:.8f}\n')
                res_file.write(f'Operation: ∫|dvz/dt|dt :-> {idvz:.8f}\n')
            
            # close the figures
            pyplot.close('all')

    # analyse the summaries
    conditions = ['predio', 'predio off', 'predio net', 'predio mean', 'predio net 5', 'predio mean 5', 'predio net 20', 'predio mean 20',
        'oito', 'oito off', 'oito net', 'oito mean', 'oito net 5', 'oito mean 5', 'oito net 20', 'oito mean 20']
    print('Analysing summaries')
    with open(os.path.join('results', 'summary.txt'), 'w') as res_file:
        res_file.write('\n')
        res_file.write('=========================\n')
        res_file.write('======== SUMMARY ========\n')
        res_file.write('=========================\n')
        for condition in conditions:
            res_file.write('\n')
            res_file.write(f'==== {condition.upper()} ====\n')
            conds = condition.split() + [None] * (3 - len(condition.split()))
            smpubt, smpredt, sactt, smdiff, sidvx, sidvz = [], [], [], [], [], []
            for _, path, method, thresh, mpubt, mpredt, actt, mdiff, idvx, idvz in summaries:
                if path == conds[0] and (method == conds[1] or conds[1] == None) and (thresh == conds[2] or conds[2] == None):
                    smpubt.append(mpubt); smpredt.append(mpredt); sactt.append(actt); smdiff.append(mdiff); sidvx.append(idvx); sidvz.append(idvz)
            res_file.write(f'Number of samples :-> {len(smpubt)}\n')
            res_file.write(f'Mean publishing time :-> {np.mean(smpubt):.4f} ± {np.std(smpubt):.4f}\n')
            res_file.write(f'Mean prediction time :-> {np.mean(smpredt):.4f} ± {np.std(smpredt):.4f}\n') 
            res_file.write(f'Acting percentage :-> {np.mean(sactt):.4f} ± {np.std(sactt):.4f}\n') 
            res_file.write(f'Mean difference :-> {np.mean(smdiff):.4f} ± {np.std(smdiff):.4f}\n') 
            res_file.write(f'Operation: ∫|dvx/dt|dt :-> {np.mean(sidvx):.4f} ± {np.std(sidvx):.4f}\n') 
            res_file.write(f'Operation: ∫|dvz/dt|dt :-> {np.mean(sidvz):.4f} ± {np.std(sidvz):.4f}\n')
        res_file.write('\n')
        res_file.write('====================================\n')
        res_file.write('======== INDIVIDUAL RESULTS ========\n')
        res_file.write('====================================\n')
        for name, path, method, thresh, mpubt, mpredt, actt, mdiff, idvx, idvz in summaries:
            res_file.write('\n')
            res_file.write(f'Name :-> {name}\n')
            res_file.write(f'Path :-> {path}\n')
            res_file.write(f'Method :-> {method}\n')
            res_file.write(f'Threshold :-> {thresh}\n')
            res_file.write(f'Mean publishing time :-> {mpubt:.4f}\n')
            res_file.write(f'Mean prediction time :-> {mpredt:.4f}\n')
            res_file.write(f'Acting percentage :-> {actt:.4f}\n')
            res_file.write(f'Mean difference :-> {mdiff:.4f}\n')
            res_file.write(f'Operation: ∫|dvx/dt|dt :-> {idvx:.4f}\n')
            res_file.write(f'Operation: ∫|dvz/dt|dt :-> {idvz:.4f}\n')

# if it is the main file
if __name__ == '__main__':
    main()
