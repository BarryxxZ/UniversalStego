# --------------------------------------------------------
# Universal Low Bit-Rate Speech Steganalysis
# Licensed under The MIT License
# Code written by Yiqin Qiu
# --------------------------------------------------------

import os
import ast
import torch
import time
import argparse
import numpy as np
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from data import get_dataloaders
from utils import set_seed
from models import MatchingIdentiModel, ContentAlignModel, CombineNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train_epoch(train_loader, model, optim, device):
    train_loss = 0
    num_correct = 0
    num_total = 0
    step = 0
    criterion = nn.CrossEntropyLoss().to(device)

    model.train()
    start = time.perf_counter()
    for batch, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
        batch_size = batch_x1.size(0)
        num_total += batch_size

        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)

        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, _ = model(batch_x1, batch_x2)
        batch_loss = criterion(batch_out, batch_y)

        _, batch_pred = nn.LogSoftmax(dim=1)(batch_out).max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        train_loss += (batch_loss.item() * batch_size)

        if batch % 50 == 0:
            step += 1
            print('\r' + '\033[32m[{:d}/{:.0f}]'.format(step, (args.train_num / args.batch_size / 50)).rjust(13) +
                  '{:.4f}'.format((num_correct / num_total)).rjust(13) + '{:.4f}\033[0m'.format(
                batch_loss.item()).rjust(18), end='')

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    epoch_time = time.perf_counter() - start
    train_loss /= num_total
    train_accuracy = (num_correct / num_total)

    return train_loss, train_accuracy, epoch_time


def evaluate_epoch(dev_loader, model, device):
    eval_loss = 0
    num_correct = 0
    num_total = 0
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for batch, (batch_x1, batch_x2, batch_y) in enumerate(dev_loader):
            batch_size = batch_x1.size(0)
            num_total += batch_size

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out, _ = model(batch_x1, batch_x2)

            batch_loss = criterion(batch_out, batch_y)
            _, batch_pred = nn.LogSoftmax(dim=1)(batch_out).max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

            eval_loss += (batch_loss.item() * batch_size)

    eval_loss /= num_total
    eval_accuracy = (num_correct / num_total)

    return eval_loss, eval_accuracy


def test(test_loader, model, model_path, device):
    model.load_state_dict(torch.load(model_path))
    cm = np.zeros((2, 2), dtype=int)
    model.eval()

    with torch.no_grad():
        num_correct = 0
        num_total = 0

        for batch, (batch_x1, batch_x2, batch_y) in enumerate(test_loader):
            batch_size = batch_x1.size(0)
            num_total += batch_size

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out, _ = model(batch_x1, batch_x2)
            _, batch_pred = nn.LogSoftmax(dim=1)(batch_out).max(dim=1)

            cm += confusion_matrix(batch_y.cpu(), batch_pred.cpu(), labels=[0, 1])
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

        test_accuracy = (num_correct / num_total)
        tn, fp, fn, tp = cm.ravel()
        fpr, fnr = fp / (fp + tn), fn / (fn + tp)

    print('[INFO] test accuracy: {:.2f}% | FPR: {:.2f}% | FNR: {:.2f}%'.format(test_accuracy * 100, fpr * 100, fnr * 100))

    return test_accuracy, fpr, fnr


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        out1, out2 = output
        self.features = out2.cpu()

    def remove(self):
        self.hook.remove()


class MyModel:
    """Model Main Class

    Args:
        arg: argument space
        train_loader: train loader
        val_loader: validation loader
        test_loader: test loader
        path_identi: model weight path of MIN
        path_align: model weight path of CAN
        path_comb: model weight path of the entire network
        device: device

    """
    def __init__(self, arg, train_loader, val_loader, test_loader, path_identi, path_align, path_comb, device):
        super(MyModel, self).__init__()
        self.arg = arg
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.path_identi = path_identi
        self.path_align = path_align
        self.path_comb = path_comb

        self.MIN = MatchingIdentiModel().to(device)
        self.CAN = ContentAlignModel().to(device)
        self.model = None

        self.opt_CAN = torch.optim.Adam(self.CAN.parameters(), lr=arg.lr_align, weight_decay=arg.weight_decay_align)

        self.opt_MIN = torch.optim.Adam(self.MIN.parameters(), lr=arg.lr_identify, weight_decay=arg.weight_decay_identify)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_MIN, milestones=[30, 70], gamma=0.5)

        self.opt_COM = None

    def train_stage_one(self):
        """the first stage of training strategy

        Pre-trianing Matching Identification Network

        """
        best_acc = 0.5
        print('   epoch    train_acc    train_loss    valid_acc    valid_loss    test_acc     dur')
        print('--------  -----------  ------------  -----------  ------------  ----------  ------')

        for epoch in range(self.arg.epoch_identify):
            running_loss, train_accuracy, train_time = train_epoch(self.train_loader, self.MIN, self.opt_MIN, self.device)
            valid_loss, valid_accuracy = evaluate_epoch(self.val_loader, self.MIN, self.device)
            test_loss, test_accuracy = evaluate_epoch(self.test_loader, self.MIN, self.device)

            self.scheduler.step()

            print('\r' + '{:d}'.format(epoch + 1).rjust(8) + '{:.4f}'.format(train_accuracy).rjust(13) +
                  '{:.4f}'.format(running_loss).rjust(14) + '{:.4f}'.format(valid_accuracy).rjust(13) +
                  '{:.4f}'.format(valid_loss).rjust(14) + '{:.4f}'.format(test_accuracy).rjust(12) +
                  '{:.0f}'.format(train_time).rjust(8), end='\n')

            if valid_accuracy >= best_acc:
                best_acc = valid_accuracy
                torch.save(self.MIN.state_dict(), self.path_identi)

    def train_stage_two(self):
        """the second stage of training strategy

        Pre-trianing Content Alignment Network

        """
        best_acc = 0.5
        print('   epoch    train_acc    train_loss    valid_acc    valid_loss    test_acc     dur')
        print('--------  -----------  ------------  -----------  ------------  ----------  ------')

        for epoch in range(self.arg.epoch_align):
            running_loss, train_accuracy, train_time = train_epoch(self.train_loader, self.CAN, self.opt_CAN, self.device)
            valid_loss, valid_accuracy = evaluate_epoch(self.val_loader, self.CAN, self.device)
            test_loss, test_accuracy = evaluate_epoch(self.test_loader, self.CAN, self.device)

            print('\r' + '{:d}'.format(epoch + 1).rjust(8) + '{:.4f}'.format(train_accuracy).rjust(13) +
                  '{:.4f}'.format(running_loss).rjust(14) + '{:.4f}'.format(valid_accuracy).rjust(13) +
                  '{:.4f}'.format(valid_loss).rjust(14) + '{:.4f}'.format(test_accuracy).rjust(12) +
                  '{:.0f}'.format(train_time).rjust(8), end='\n')

            if valid_accuracy >= best_acc:
                best_acc = valid_accuracy
                torch.save(self.CAN.state_dict(), self.path_align)

    def train_stage_three(self, arg):
        """the third stage of training strategy

        Fine-tuning the entire network

        """
        self.model = CombineNet(self.path_identi, self.path_align).to(self.device)
        summary(self.model, input_size=([(arg.batch_size, int(float(arg.length) * 50), 53),
                                               (arg.batch_size, int(float(arg.length) * 50), 53)]), dtypes=[torch.int, torch.int])
        self.opt_COM = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.arg.lr_comb, weight_decay=self.arg.weight_decay_comb)

        best_acc = 0.5
        print('   epoch    train_acc    train_loss    valid_acc    valid_loss    test_acc     dur')
        print('--------  -----------  ------------  -----------  ------------  ----------  ------')

        for epoch in range(self.arg.epoch_align):
            running_loss, train_accuracy, train_time = train_epoch(self.train_loader, self.model, self.opt_COM, self.device)
            valid_loss, valid_accuracy = evaluate_epoch(self.val_loader, self.model, self.device)
            test_loss, test_accuracy = evaluate_epoch(self.test_loader, self.model, self.device)

            print('\r' + '{:d}'.format(epoch + 1).rjust(8) + '{:.4f}'.format(train_accuracy).rjust(13) +
                  '{:.4f}'.format(running_loss).rjust(14) + '{:.4f}'.format(valid_accuracy).rjust(13) +
                  '{:.4f}'.format(valid_loss).rjust(14) + '{:.4f}'.format(test_accuracy).rjust(12) +
                  '{:.0f}'.format(train_time).rjust(8), end='\n')

            if valid_accuracy >= best_acc:
                best_acc = valid_accuracy
                torch.save(self.model.state_dict(), self.path_comb)

    def test_model(self, test_loader, device, model_path=None):
        """test model

        Args:
            test_loader: test loader
            device: device
            model_path: model weight path, use default path if not assign

        Returns:
            test_accuracy: test accuracy
            fpr: test false positive rate
            fnr: test false negative rate

        """
        if model_path is None:
            model_path = self.path_comb
        if self.model is None:
            self.model = CombineNet(self.path_identi, self.path_align).to(self.device)
        test_accuracy, fpr, fnr = test(test_loader, self.model, model_path, device)
        return test_accuracy, fpr, fnr

    @torch.no_grad()
    def forensic(self, domain_loader, device, model_path=None):
        """forensic stego domain

        Args:
            domain_loader: domain loader, please define a dataloader for yourself that returns the domain label
            device: device
            model_path: model weight path, use default path if not assign

        Returns:
            test_accuracy: forensic accuracy

        """
        if model_path is None:
            model_path = self.path_comb
        if self.model is None:
            self.model = CombineNet(self.path_identi, self.path_align).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        num_correct = 0
        num_total = 0
        for batch, (batch_x1, batch_x2, batch_y1, batch_y2) in enumerate(domain_loader):
            batch_size = batch_x1.size(0)

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_y1 = batch_y1.view(-1).type(torch.int64).to(device)
            batch_y2 = batch_y2.view(-1).type(torch.int64).to(device)
            # get matching scores via hook class
            scores_out = LayerActivations(self.model.identi_net.domain_match, 'domain_match')
            batch_out, _ = self.model(batch_x1, batch_x2)
            _, batch_pred = nn.LogSoftmax(dim=1)(batch_out).max(dim=1)

            scores_out.remove()
            scores = scores_out.features
            # get predict domain label via forensic algorithm
            domain_pred, _ = torch.mode(scores[:, :, :3].argmax(dim=-1)[:])

            for pos in range(batch_size):
                # only calculate in stego samples that been correctly detected
                if batch_pred.cpu()[pos] == batch_y1.cpu()[pos]:
                    num_total += 1
                    if domain_pred.cpu()[pos] == batch_y2.cpu()[pos]:
                        num_correct += 1

        test_accuracy = (num_correct / num_total)
        print('[INFO] domain forensic accuracy: {:.2f}%'.format(test_accuracy * 100))

        return test_accuracy


def main(arg):
    torch.cuda.set_device(0)
    if arg.seed is not None:
        set_seed(arg.seed)

    train_loader, val_loader, test_loader, domain_loader = get_dataloaders(arg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Device: {}'.format(device))

    path_align = './weights/align_{}s_{}.pth'.format(args.length, args.em_rate)
    path_identify = './weights/identify_{}s_{}.pth'.format(args.length, args.em_rate)
    path_comb = './weights/comb_{}s_{}.pth'.format(args.length, args.em_rate)

    ours_model = MyModel(arg, train_loader, val_loader, test_loader, path_identify, path_align, path_comb, device)

    print('[INFO] Pre-training identification network')
    print('[INFO] identification network path: {}'.format(path_identify))

    if arg.train_identify:
        summary(ours_model.MIN, input_size=([(arg.batch_size, int(float(arg.length) * 50), 53),
                                             (arg.batch_size, int(float(arg.length) * 50), 53)]), dtypes=[torch.int, torch.int])
        ours_model.train_stage_one()

    print('[INFO] Pre-training aligned network')
    print('[INFO] aligned network path: {}'.format(path_align))

    if arg.train_align:
        summary(ours_model.CAN, input_size=([(arg.batch_size, int(float(arg.length) * 50), 53),
                                             (arg.batch_size, int(float(arg.length) * 50), 53)]), dtypes=[torch.int, torch.int])
        ours_model.train_stage_two()

    print('[INFO] Fine-tuning combination network')
    print('[INFO] combination network path: {}'.format(path_comb))

    if arg.train_comb:
        ours_model.train_stage_three(arg)

    if arg.add_log:
        test_accuracy, fpr, fnr = ours_model.test_model(test_loader, device)
        forensic_accuracy = ours_model.forensic(domain_loader, device)
        with open(arg.log_dir, 'a') as f:
            f.writelines(["\n" + path_comb + " Accuracy %0.2f  " % (test_accuracy * 100) + "FPR %0.2f  " % (fpr * 100)
                          + "FNR %0.2f  " % (fnr * 100) + "Forensic_Accuracy %0.2f  " % (forensic_accuracy * 100)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universe Detection')
    parser.add_argument('--length', type=str, default='1.0')
    parser.add_argument('--em_rate', type=str, default='30')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epoch_align', type=int, default=100)
    parser.add_argument('--lr_align', type=float, default=0.001)
    parser.add_argument('--weight_decay_align', type=float, default=0.0001)

    parser.add_argument('--epoch_identify', type=int, default=100)
    parser.add_argument('--lr_identify', type=float, default=0.0002)
    parser.add_argument('--weight_decay_identify', type=float, default=0.001)

    parser.add_argument('--epoch_comb', type=int, default=50)
    parser.add_argument('--lr_comb', type=float, default=0.0001)
    parser.add_argument('--weight_decay_comb', type=float, default=0)

    parser.add_argument('--train_num', type=int, default=72000)
    parser.add_argument('--val_num', type=int, default=12000)
    parser.add_argument('--test_num', type=int, default=12000)

    parser.add_argument('--train_align', type=ast.literal_eval, default=True)
    parser.add_argument('--train_identify', type=ast.literal_eval, default=True)
    parser.add_argument('--train_comb', type=ast.literal_eval, default=True)

    parser.add_argument('--add_log', type=ast.literal_eval, default=True)
    parser.add_argument('--log_dir', type=str, default='./logs/result.txt')
    parser.add_argument('--test_mode', type=int, default=0)

    args = parser.parse_args()

    main(args)
