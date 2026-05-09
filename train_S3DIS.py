#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on S3DIS dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import argparse
import json
import platform
import signal
import os
import sys
import time
from os import makedirs
from os.path import exists

# Dataset
from datasets.S3DIS import *
from torch.utils.data import DataLoader
import torch

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class S3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # Define layers
    # architecture = ['simple',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb',
    #                 'resnetb',
    #                 'nearest_upsample',
    #                 'unary',
    #                 'nearest_upsample',
    #                 'unary',
    #                 'nearest_upsample',
    #                 'unary',
    #                 'nearest_upsample',
    #                 'unary']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Radius of the input sphere (decrease value to reduce memory cost)
    in_radius = 1.2

    # Size of the first subsampling grid in meter (increase value to reduce memory cost)
    first_subsampling_dl = 0.03

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 5

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 400

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch (decrease to reduce memory cost, but it should remain > 3 for stability)
    batch_num = 6

    # Number of steps per epochs
    epoch_steps = 50

    # Number of validation examples per epoch
    validation_size = 10

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8

    # Experiment switches for ablation studies
    use_attention_gate = True
    loss_type = 'ce'

    # Area_5 is kept as validation/test by default. Area_7 can be mixed into
    # training at a low frequency for non-official extension experiments.
    validation_area = 'Area_5'
    include_area7 = False
    area7_sampling_ratio = 10

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Validation area index: 0=Area_1, 1=Area_2, ..., 4=Area_5 (default)
    validation_split = 4

    # Do we nee to save convergence
    saving = True
    saving_path = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train KPFCNN on S3DIS with ablation switches.')
    parser.add_argument('legacy_saving_path', nargs='?', default=None,
                        help='Backward-compatible positional saving path.')
    parser.add_argument('--saving-path', default=None, help='Directory where logs and checkpoints are saved.')
    parser.add_argument('--epochs', type=int, default=None, help='Override max_epoch.')
    parser.add_argument('--attention', choices=['on', 'off'], default=None,
                        help='Enable or disable AttentionGate skip filtering.')
    parser.add_argument('--loss', choices=['ce', 'weighted_ce'], default=None,
                        help='Loss type for segmentation logits.')
    parser.add_argument('--include-area7', choices=['on', 'off'], default=None,
                        help='Whether to include Area_7 in the training split.')
    parser.add_argument('--area7-ratio', type=int, default=None,
                        help='Official training areas : Area_7 sampling ratio, e.g. 10 means 10:1.')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value.')
    parser.add_argument('--previous-training-path', default=None,
                        help='Optional result directory to restore from. Defaults to a fresh training.')
    parser.add_argument('--checkpoint-index', type=int, default=None,
                        help='Checkpoint index to restore. Omit to use current_chkp.tar.')
    return parser.parse_args()


def compute_class_weights(dataset):
    """Compute inverse-square-root class weights from the selected training split."""
    counts = np.zeros(dataset.num_classes, dtype=np.float64)
    for labels in dataset.input_labels:
        for i, label_value in enumerate(dataset.label_values):
            counts[i] += np.sum(labels == label_value)

    counts = np.maximum(counts, 1.0)
    proportions = counts / np.sum(counts)
    weights = np.sqrt(1.0 / proportions)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32).tolist(), counts.astype(np.int64).tolist()


def write_run_info(config, training_dataset, args, class_counts=None):
    if not config.saving or config.saving_path is None:
        return

    if not exists(config.saving_path):
        makedirs(config.saving_path)

    area7_percent = 0.0
    if config.include_area7:
        area7_percent = 100.0 / (float(config.area7_sampling_ratio) + 1.0)

    info = {
        'experiment_id': os.path.basename(os.path.normpath(config.saving_path)),
        'epochs': config.max_epoch,
        'use_attention_gate': bool(config.use_attention_gate),
        'loss_type': config.loss_type,
        'include_area7': bool(config.include_area7),
        'area7_sampling_ratio': int(config.area7_sampling_ratio),
        'area7_expected_sampling_percent': round(area7_percent, 3),
        'effective_train_areas': training_dataset.cloud_names,
        'validation_area': config.validation_area,
        'batch_num': config.batch_num,
        'epoch_steps': config.epoch_steps,
        'validation_size': config.validation_size,
        'learning_rate': config.learning_rate,
        'checkpoint_gap': config.checkpoint_gap,
        'class_counts': class_counts,
        'class_w': [float(w) for w in config.class_w],
        'gpu': args.gpu,
        'python': sys.version.replace('\n', ' '),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'platform': platform.platform(),
        'command': ' '.join(sys.argv),
    }

    with open(os.path.join(config.saving_path, 'run_info.md'), 'w') as f:
        f.write('# S3DIS Experiment Run Info\n\n')
        for key, value in info.items():
            f.write('- {}: {}\n'.format(key, value))

    with open(os.path.join(config.saving_path, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    args = parse_args()

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = args.gpu

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    previous_training_path = args.previous_training_path

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = args.checkpoint_index
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join(previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = S3DISConfig()
    if previous_training_path:
        config.load(previous_training_path)

    # Apply command-line overrides
    saving_path = args.saving_path or args.legacy_saving_path
    if saving_path is not None:
        config.saving_path = saving_path
    if args.epochs is not None:
        config.max_epoch = args.epochs
    if args.attention is not None:
        config.use_attention_gate = args.attention == 'on'
    if args.loss is not None:
        config.loss_type = args.loss
    if args.include_area7 is not None:
        config.include_area7 = args.include_area7 == 'on'
    if args.area7_ratio is not None:
        if args.area7_ratio <= 0:
            raise ValueError('--area7-ratio must be positive')
        config.area7_sampling_ratio = args.area7_ratio

    config.lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, config.max_epoch)}

    # Initialize datasets
    training_dataset = S3DISDataset(config, set='training', use_potentials=True)
    test_dataset = S3DISDataset(config, set='validation', use_potentials=True)

    if config.loss_type == 'weighted_ce':
        config.class_w, class_counts = compute_class_weights(training_dataset)
        config.segloss_balance = 'class'
    else:
        config.class_w = []
        config.segloss_balance = 'none'
        class_counts = None

    write_run_info(config, training_dataset, args, class_counts=class_counts)

    # Initialize samplers
    training_sampler = S3DISSampler(training_dataset)
    test_sampler = S3DISSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=S3DISCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=S3DISCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
