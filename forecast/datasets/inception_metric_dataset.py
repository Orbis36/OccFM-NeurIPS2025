import numpy as np
import glob
from torch.utils.data import Dataset
import re

class FVDEval(Dataset):
    def __init__(self, root_path, **kwargs):
        super().__init__()

        self.all_gt_files = self.sort_files(glob.glob(root_path+'gt*'))
        self.all_pred_files = self.sort_files(glob.glob(root_path+'pred*'))

        assert len(self.all_gt_files) == len(self.all_pred_files)

    @staticmethod
    def sort_files(file_list):
        idx = [x.split('/')[-1] for x in file_list]
        idx = [int(re.search(r'(?:pred|gt)_([0-9]+)', x).group(1)) for x in idx]
        new_idx = [None] * len(idx)
        for idx_ori, sort_idx in enumerate(idx):
            new_idx[sort_idx] = file_list[idx_ori]
        return new_idx

    def __len__(self):
        return len(self.all_gt_files)

    def __getitem__(self, item):

        gt_path, pred_path = self.all_gt_files[item], self.all_pred_files[item]
        gt = np.load(gt_path)
        pred = np.load(pred_path)
        return gt, pred