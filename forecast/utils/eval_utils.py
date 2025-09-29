import torch
import torch.distributed as dist
import numpy as np

class DistributedListBuffer:
    """
    进程内维护一个 list；gather() 时一次性把所有 rank 的 list 合并。
    """
    def __init__(self):
        self._local = []

    def append(self, item):
        """item 可以是任意可 pickle 对象（dict / ndarray …）"""
        self._local.append(item)

    def gather(self):
        """
        所有进程都要调用。返回：
            - rank 0: 合并后的大列表
            - 其余  : None
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._local   # 单卡直接返回

        world_size = dist.get_world_size()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, self._local)
        if dist.get_rank() == 0:
            merged = [item for sub in gathered for item in sub]
            return merged
        else:
            return None

class DistributedDictMeanCounter:
    """
    并行统计一组标量，支持 DDP 全局平均。
    """

    def __init__(self, device=None):
        self.device = device
        self.reset()

    # --------------------------------------
    def reset(self):
        self._sum   = {}      # key → tensor(sum)
        self._count = {}      # key → tensor(count)

    # --------------------------------------
    @torch.inference_mode()
    def update(self, value_dict: dict):
        """
        value_dict: {key: scalar | 0-D tensor}
        每个 batch 调一次。
        """
        for k, v in value_dict.items():
            if not torch.is_tensor(v):
                v = torch.tensor(v, dtype=torch.float32, device=self.device)
            else:
                v = v.to(self.device, dtype=torch.float32)

            if k not in self._sum:        # 第一次见到 key
                self._sum[k]   = torch.zeros((), device=self.device)
                self._count[k] = torch.zeros((), device=self.device)

            self._sum[k]   += v
            self._count[k] += 1.0

    # --------------------------------------
    @torch.inference_mode()
    def compute(self) -> dict:
        """
        在所有进程都调用，返回 {key: global_mean}
        """
        if not self._sum:          # 从未 update 过
            return {}

        keys = sorted(self._sum.keys())

        sums   = torch.stack([self._sum[k]   for k in keys])
        counts = torch.stack([self._count[k] for k in keys])

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sums,   op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        means = sums / torch.clamp_min(counts, 1.0)
        return {k: means[i].item() for i, k in enumerate(keys)}

class multi_step_MeanIou:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 times=1, dist=True):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        self.dist = dist

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.total_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.total_positive = torch.zeros(self.times, self.num_classes).cuda()

    def _after_step(self, outputses, targetses):

        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:, t, ...][targetses[:, t, ...] != self.ignore_label].cuda()
            targets = targetses[:, t, ...][targetses[:, t, ...] != self.ignore_label].cuda()
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c) & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()

    def _after_epoch(self):

        if self.dist:
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
        mious = []
        raw_ious_per_frame = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            raw_ious_per_frame.append(ious)
            miou = np.mean(ious)
            mious.append(miou * 100)
        return mious, np.asarray(raw_ious_per_frame)