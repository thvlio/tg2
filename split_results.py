# import necessary packages
import subprocess
import argparse
import os

def split_results(args):
    ''' Splits results from .bag file given. '''

    # create subprocess to run
    print('Running subprocess')
    if args['track'] == 'predio':
        bag = args['bag']
        dest = os.path.join(os.path.split(args['bag'])[0], f"{args['name']}_{args['track']}_{args['method']}.bag")
        t0 = args['start'] + args['timestamps'][0]
        t1 = args['start'] + args['timestamps'][1]
        subprocess.check_call(['rosbag', 'filter', f'{bag}', f'{dest}', f"t.secs >= {t0} and t.secs <= {t1} and (topic == \'info\' or topic == \'pose\')"])
    else:
        bag = args['bag']
        dest = os.path.join(os.path.split(args['bag'])[0], f"{args['name']}_{args['track']}_{args['method']}.bag")
        t0 = args['start'] + args['timestamps'][0]
        t1 = args['start'] + args['timestamps'][1]
        t2 = args['start'] + args['timestamps'][2]
        t3 = args['start'] + args['timestamps'][3]
        subprocess.check_call(['rosbag', 'filter', f'{bag}', f'{dest}', f"((t.secs >= {t0} and t.secs <= {t1}) or (t.secs >= {t2} and t.secs <= {t3})) and (topic == \'info\' or topic == \'pose\')"])

def main():
    ''' Main function. '''

    # setup the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name', type=str,
        help='name of the tester')
    ap.add_argument('-t', '--track', type=str,
        help='track used')
    ap.add_argument('-m', '--method', type=str,
        help='method used')
    ap.add_argument('-b', '--bag', type=str,
        help='path to bag file')
    ap.add_argument('-s', '--start', type=int,
        help='start of bag file')
    ap.add_argument('-ts', '--timestamps', type=int, nargs='+',
        help='list of timestamps')
    args = vars(ap.parse_args())

    # split the results
    split_results(args)

# if it is the main file
if __name__ == '__main__':
    main()
