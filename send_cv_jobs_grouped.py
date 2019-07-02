# import necessary packages
import subprocess
import os

# import files
import config

def main():
    ''' Main function. '''

    # dictionaries with the job and its output
    jobs_stdout = {}
    jobs_stderr = {}

    # create folder inside jobs
    if not os.path.isdir('jobs/{}'.format(config.dataset_name)):
        os.makedirs('jobs/{}'.format(config.dataset_name))

    # loop on different architectures
    for epochs, batch_size, layer_units in config.network_archs:

        # model string
        model_string = '{}'.format(
            '_'.join([
                'L' + '-'.join([str(units) for units in layer_units]),
                'B' + str(batch_size)
            ])
        )
        print('Model: {}'.format(model_string))

        # writing instructions on the job file
        print('\tWriting job file')
        job_name = '{}_CV'.format(model_string)
        with open('jobs/{}/{}'.format(config.dataset_name, job_name), 'w') as job_file:
            job_file.write('#PBS -N {j} -e jobs/{f}/{j}.e.txt -o jobs/{f}/{j}.o.txt -l nodes=1:ppn=2\n'.format(f=config.dataset_name, j=job_name))
            job_file.write('cd $PBS_O_WORKDIR\n')
            job_file.write('python cv_train.py --dataset {d} --epochs {e} --batch-size {bs} --layer-units {lu} --cpu\n'.format(
                d=config.dataset_name, e=epochs, bs=batch_size, lu=' '.join([str(u) for u in layer_units])
            ))

        # running the command to queue the job
        print('\tRunning job file')
        job_id = subprocess.Popen(['qsub', 'jobs/{}/{}'.format(config.dataset_name, job_name)],
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)

        # save the command output on the dictionaries
        print('\tSaving output')
        stdout, stderr = job_id.communicate()
        jobs_stdout[job_name] = stdout.decode() if stdout is not None else None
        #jobs_stderr[job_name] = stderr.decode() if stderr is not None else None

    # write the output to a file
    print('Writing jobs dictionary to file')
    with open('jobs/{}/train_jobs_dict'.format(config.dataset_name), 'w') as job_dict_file:
        for k, v in jobs_stdout.items():
            job_dict_file.write('{}: {}'.format(k, v))

# if it is the main file
if __name__ == '__main__':
    main()
