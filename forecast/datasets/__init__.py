from .dataset import DatasetTemplate
from .nuscenes_dataset import NuScenesDataset
from functools import partial

from forecast.utils import common_utils

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'NuScenesDataset': NuScenesDataset,
}


def build_dataloader(dataset_cfg, batch_size, num_workers, cache_mode, training=True, seed=None, rank=None, world_size=None):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=common_utils.lowercase_keys(dataset_cfg), batch_size=batch_size, training=training, cache_mode=cache_mode
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training, drop_last=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, collate_fn=dataset.collate_batch,
        sampler=sampler, drop_last=True, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    return dataset, dataloader

def reset_batch_size(data_loader, new_batch_size, rank, world_size, training=False):
    sampler = DistributedSampler(data_loader.dataset, num_replicas=world_size,
                                 rank=rank, shuffle=training)
    return DataLoader(
        dataset=data_loader.dataset,
        batch_size=new_batch_size,
        shuffle=False, sampler=sampler,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
        drop_last=False,
        collate_fn=data_loader.collate_fn,
    )