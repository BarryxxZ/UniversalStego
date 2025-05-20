import torch
from torch.utils.data import Dataset


def read_file(file_path):
    """read coding parameters from file

    Args:
        file_path (str): file path

    Returns:
        array: array of coding parameters

    """
    file = open(file_path, 'r')
    lines = file.readlines()
    array = []
    flag = 0
    # cover param used to mask
    mask = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for line in lines:
        row = [int(item) for item in line.split(' ')]
        flag += 1
        # use cover param to mask the first subframe of pulse positions
        if flag == 1:
            for pos in range(5, 15):
                row[pos] = mask[pos - 5]
        for frac in range(-4, 0):
            row[frac] = row[frac] + 2
        array.append(row)
    file.close()
    return array


class LoadData(Dataset):
    def __init__(self, file_list, dual_mode=False):
        self.file_list = file_list
        self.dual_mode = dual_mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x, y = self.file_list[index]
        if self.dual_mode:
            x1, x2 = x
            x1 = read_file(x1)
            x2 = read_file(x2)
            x1 = torch.LongTensor(x1)
            x2 = torch.LongTensor(x2)
            return x1, x2, y
        else:
            x = read_file(x)
            x = torch.LongTensor(x)
            return x, y


class LoadDataDomain(Dataset):
    def __init__(self, file_list, dual_mode=False):
        self.file_list = file_list
        self.dual_mode = dual_mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x, y1, y2 = self.file_list[index]
        if self.dual_mode:
            x1, x2 = x
            x1 = read_file(x1)
            x2 = read_file(x2)
            x1 = torch.LongTensor(x1)
            x2 = torch.LongTensor(x2)
            return x1, x2, y1, y2
        else:
            x = read_file(x)
            x = torch.LongTensor(x)
            return x, y1, y2
