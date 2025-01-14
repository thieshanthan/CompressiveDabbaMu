import shutil
import os
import itertools
import glob
import shutil


def write_errormsg2file(msg, error_file_name):
    if not os.path.isfile(error_file_name):
        with open(error_file_name, 'w') as f:
            f.write(f'error : {msg}\n')
    else:
        with open(error_file_name, 'a') as f:
            f.write(f'error : {msg}\n')


def safe_do_exps(exps_dict=None, general_opts=None, device=None,
                 exp_dir='../figs/test', save_special=False, count_only=True,
                 save_dir_special_root=None, run=None):
    exp_idx = 0
    keys = list(exps.keys())

    val_list_list = []
    key_list = []  # eg: 'MODEL.MODEL_A.rotation_lambda'
    key_suffix_list = []  # eg: 'rotation_lambda'

    for key, val_list in exps_dict.items():
        key_list.append(key)
        key_suffix_list.append(key.split('.')[-1])

        val_list_list.append(val_list)

    attr_combination_list = [list(s) for s in itertools.product(*val_list_list)]

    print(f'number of total experiments : {len(attr_combination_list)}')

    count_already_trained = 0
    count_train_from_begining = 0
    count_deleted = 0
    for attr_combination in attr_combination_list:
        save_dir = f'{exp_dir}/'
        opts = []
        for idx in range(len(attr_combination)):
            opts += [key_list[idx], attr_combination[idx]]
            attr = attr_combination[idx]

            #####
            attr_is_list = False
            try:
                attr_is_list = isinstance(eval(attr), list)
            except:
                pass
            if attr_is_list:
                attr = '_'.join(list(map(str, eval(
                    attr))))  ## [, ] should not be in the directory name because glob is sensitive to that !
            #####

            save_dir += f'{key_suffix_list[idx]}({attr})@'

        save_dir = save_dir[:-1]  # remove last '@'

        exp_idx += 1
        opts_other = ['NAME', f'exp_idx({exp_idx})',
                      'GENERAL.device', device,
                      'GENERAL.save_dir', save_dir
                      ]

        opts_other += general_opts

        opts = opts_other + opts

        if os.path.isdir(save_dir):
            if 'TRAIN.epochs' in opts:
                epochs_idx = opts.index('TRAIN.epochs')
                epochs = int(opts[epochs_idx + 1])  # IF THIS GIVES ERROR -> THERE IS A PROBLEM

                if len(glob.glob(f'{save_dir}/{epochs}_*.jpg')) != 0:
                    count_already_trained += 1
                    print(f'PASSING (already trained) -> {save_dir}')
                    continue
                else:
                    count_train_from_begining += 1
                    count_deleted += 1
                    print(f'deleting -> {save_dir}')
                    if not count_only:
                        shutil.rmtree(save_dir)
            else:
                count_train_from_begining += 1
                count_deleted += 1
                print(f'deleting -> {save_dir}')
                if not count_only:
                    shutil.rmtree(save_dir)
        else:
            count_train_from_begining += 1

        save_folder_name = save_dir.split('/')[-1]

        if not count_only:
            print(f'running  -> {save_dir}')
            if len(save_folder_name) > 255:
                print(
                    f'\nFolder length is too long: len(results_saving_folder) -> {len(save_folder_name)} (<= 255)')
                print(save_folder_name)

            # run(opts= opts, save_special=save_special)

            try:
                if save_dir_special_root == None:
                    save_dir_special_root = exp_dir

                save_dir_special = f'{save_dir_special_root}/{save_folder_name}'
                run(opts=opts, save_special=save_special,
                    save_dir_special=save_dir_special)
            except Exception as e:
                error_file_name = f'{exp_dir}/errors.txt'
                write_errormsg2file(f'ERROR : {save_dir}\n {e} \n\n',
                                    error_file_name)
                print(f'ERROR : {save_dir}\n {e} \n\n')

    print(f'\n\nCOUNT ONLY (no exps running/ deleting) : {count_only}')
    print('count_already_trained (tot_epochs completed): ',
          count_already_trained)
    print('count_train_from_begining : ', count_train_from_begining)
    print('count_train_from_begining (after deleting existing exp) : ',
          count_deleted)


###################### CREATE DIRECTORIES


exp_dir = './figs/sample_exp'
save_dir_special_root = None

if not os.path.isdir('./figs'): os.mkdir('./figs')
try:
    shutil.rmtree(exp_dir)
except:
    pass
try:
    shutil.rmtree(save_dir_special_root)
except:
    pass

os.mkdir(exp_dir)
if save_dir_special_root != None: os.mkdir(save_dir_special_root)
######################### SET PARAMETERS
##

import argparse

parser = argparse.ArgumentParser(description='Differentiable Microscopy')
parser.add_argument('--exp_type', type=str,
                    help='exp type: contentaware or segmentation',
                    default='contentaware')
parser.add_argument('--dataset_name', type=str, help='dataset_name',
                    default='mnistdigits_grid2patch')
parser.add_argument('--data_dir', type=str, help='dataset_name',
                    default='./datasets/patchmnist')
args = parser.parse_args()

exp_type = args.exp_type
dataset_name = args.dataset_name
data_dir = args.data_dir
##

count_only = False

device = 'cuda:0'

exps = {
    'DATASET.name': [dataset_name],  # 'confocal'

    'MODEL.MODEL_A.lambda_scale_factor': ['2'],  # '8'
    'MODEL.MODEL_H.T': ['4'],
    'MODEL.MODEL_DECODER.upsample_net': ['custom_v2'],
    'MODEL.MODEL_H.lr_H': ['1.0'],

    # change this for for PSF experiments: #Put exPSF, sPSF(=emPSF) that is defined in the modules/psfs.py or modules/kernels.py
    'MODEL.MODEL_A.sPSF': ['impulse(side_len=5)'],  # 'confocal_w1_emPSF()'
    'MODEL.MODEL_A.exPSF': ['impulse(side_len=5)'],  # 'confocal_w1_exPSF()'

    # change this for noise experiments
    'MODEL.MODEL_A.readnoise_std': ['0.0'],
    # readnoise standard deviation (e.g.: '6.0')
    'MODEL.MODEL_A.rotation_lambda': ['10000.0'],  # photon count (e.g.: '10.0')

    'DATASET.img_size': ['32'],  # 256
    'DATASET.num_samples_train': ['50']}  # 3000

general_opts = ['TRAIN.show_results_epoch', '1',  # 5
                'TRAIN.epochs', '2',  # 150
                'DATASET.data_dir', data_dir]

####################### TRAIN AND TEST MODELS

from train import run as run_contentaware
from train_segmentation import run as run_segmentation

if exp_type == 'contentaware':
    run = run_contentaware
elif exp_type == 'segmentation':
    run = run_segmentation
else:
    raise NotImplementedError

safe_do_exps(exps, general_opts, device, exp_dir=exp_dir, save_special=False,
             count_only=count_only, save_dir_special_root=save_dir_special_root,
             run=run)
