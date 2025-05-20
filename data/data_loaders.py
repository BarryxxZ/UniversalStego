# --------------------------------------------------------
# Universal Low Bit-Rate Speech Steganalysis
# Licensed under The MIT License
# Code written by Yiqin Qiu
# --------------------------------------------------------

import os

import numpy
import torch.nn

from configs import TrainConfigs, TestConfigs
from data.datasets import LoadData, LoadDataDomain
from torch.utils.data import DataLoader


def get_file_list_dual(file_dir, re_file_dir, train_num, val_num, test_num, spilt, return_domain_label=False):
    """Get file lists of samples

    Args:
        file_dir: Folder dict with file paths
        re_file_dir: Folder dict with file paths for recompressed samples
        train_num: Number of training samples
        val_num: Number of validation samples
        test_num: Number of test samples
        spilt: Length of folder dict
        return_domain_label: Return domain label for forensic dict

    Returns:
        train_list: File list of training samples
        val_list: File list of validation samples
        test_list: File list of test samples

    """
    train_list = []
    val_list = []
    test_list = []

    for dir_index in range(len(file_dir)):
        file_list = []
        all_file_list = os.listdir(file_dir[dir_index]['folder'])
        all_re_file_list = os.listdir(re_file_dir[dir_index]['folder'])

        # assign domain label according to the index of path in folder dict
        if dir_index < 6:
            domain_label = -1
        elif dir_index == 6 or dir_index == 7:
            domain_label = 1
        elif dir_index == 8 or dir_index == 9:
            domain_label = 0
        elif dir_index == 10 or dir_index == 11:
            domain_label = 2

        for index in range(len(all_file_list)):
            if not return_domain_label:
                file_list.append(([os.path.join(file_dir[dir_index]['folder'], all_file_list[index]), os.path.join(
                    re_file_dir[dir_index]['folder'], all_re_file_list[index])], file_dir[dir_index]['class']))
            else:
                file_list.append(([os.path.join(file_dir[dir_index]['folder'], all_file_list[index]), os.path.join(
                    re_file_dir[dir_index]['folder'], all_re_file_list[index])], file_dir[dir_index]['class'], domain_label))

        train_list += file_list[: train_num // spilt]
        val_list += file_list[train_num // spilt: (train_num + val_num) // spilt]
        test_list += file_list[-test_num // spilt:]

    return train_list, val_list, test_list


def get_dataloaders(arg):
    """Get dataloader objects

    Args:
        arg: Argument Namespace

    Returns:
        train_loader: train loader
        val_loader: validation loader
        test_loader: test loader
        domain_loader: loader containing domain label that only used for forensic test

    """
    # test-only mode when test_mode is not 0
    # Get samples in ['all', 'FCB/Geiser', 'LPC/CNV', 'ACB/Huang']
    if arg.test_mode == 0:
        folder_configs = TrainConfigs(arg)
    else:
        folder_configs = TestConfigs(arg)

    print('[INFO] read data file')
    train_list, val_list, test_list = get_file_list_dual(folder_configs.FOLDERS, folder_configs.RE_FOLDERS,
                                                         arg.train_num, arg.val_num, arg.test_num, len(folder_configs.FOLDERS))
    print('[INFO] train number: {}, validate number: {}, test number: {}'.format(len(train_list), len(val_list), len(test_list)))

    train_loader = DataLoader(dataset=LoadData(train_list, True), batch_size=arg.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=LoadData(val_list, True), batch_size=arg.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(dataset=LoadData(test_list, True), batch_size=arg.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    _, _, domain_list = get_file_list_dual(folder_configs.FOLDERS, folder_configs.RE_FOLDERS, arg.train_num, arg.val_num,
                                           arg.test_num, len(folder_configs.FOLDERS), True)
    domain_loader = DataLoader(dataset=LoadDataDomain(domain_list[6000:], True), batch_size=arg.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, domain_loader
