#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import *

ignore_all_other = 1
ignore_outside_main_label = 2
ignore_inside_main_label = 3 #assumes any could be a mix of benign and malignant

class DC_and_CE_loss_masked(nn.Module):

    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", weight_ce=1, weight_dice=1,
                 log_dice=False, mask_ignore_type = ignore_outside_main_label  ):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()

        self.ignore_label = None
        square_dice = False
        ce_kwargs['reduction'] = 'none'

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)



        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

        # if we are ignoring the other label we don't mask it out inside the other target label
        self.mask_type = mask_ignore_type

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        print(f'Shape: {target_shape}' )
        print(f'Target: {target}')
        mask=target!=0
        x = mask[:,0]
        print(f'Value: {x}')
        print(f'Valuse shape: {x.shape}')
        raise Exception('Just testing')
        #mask out minor label
        assert target.shape[1] == 1, 'not implemented for one hot encoding'
        seg_labels, cnts = torch.unique(target, return_counts=True)
        ignore_label=1
        keep_label=2
        if 1 not in seg_labels:
            ignore_label = 1
        elif 2 not in seg_labels:
            ignore_label = 2
        elif cnts[seg_labels==1]>cnts[seg_labels==2]:
            print('Target has labels 1 and 2')
            ignore_label = 1
            keep_label = 2
        else:
            print('Target has labels 1 and 2')
            ignore_label=2
            keep_label=1

        mask_target = target != ignore_label
        target[~mask_target] =0

        #ignore if we labeled things in targe with this label
        mask_net = net_output != ignore_label

        # ignore benign and malignant inside

        mask = torch.logical_and(mask_net, mask_target)
        if self.mask_type == ignore_all_other:
            pass
        else:
            keep_label_mask = target == keep_label
            mask = torch.logical_or(mask, keep_label_mask)
        mask = mask.float()

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class nnUNetTrainerV2_MaskedDiceCE_IgnoreOut(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, mask_ignore_type =ignore_outside_main_label)

class nnUNetTrainerV2_MaskedDiceCE_IgnoreInside(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, mask_ignore_type =ignore_outside_main_label)

