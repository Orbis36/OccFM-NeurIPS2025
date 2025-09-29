import os
import pickle
import numpy as np

from .dataset import DatasetTemplate
from nuscenes.nuscenes import NuScenes

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, batch_size, training, cache_mode):
        super().__init__(dataset_cfg, training)

        self.cache_mode = cache_mode
        self.sem_mode = dataset_cfg.sem_mode
        self.label_name = dataset_cfg.label_name
        self.sequence_length = dataset_cfg.sequence_length
        self.x_sampled = None
        self.win_size = dataset_cfg.get('win_size', 1)
        self.hist_last = dataset_cfg.get('hist_last', 6)

        pickle_path = dataset_cfg['info_path']['train' if training else 'test'][0]
        pickle_path = os.path.join(dataset_cfg['data_path'], pickle_path)
        with open(pickle_path, 'rb') as f:
            self.infos = pickle.load(f)['infos']

        if not cache_mode: # used to control dataset in VAE or CFM
            self.nuSc_context_manager = NuScenes(version='v1.0-trainval', dataroot=dataset_cfg['data_path'])

            using_scenes = list(self.infos.keys())
            init_pos = 0
            for scene in self.nuSc_context_manager.scene:
                if scene["name"] not in using_scenes:
                    continue

                all_token_with_order = [self.infos[scene["name"]][i]['token'] for i in range(len(self.infos[scene["name"]]))]
                path = [dataset_cfg['data_path'] + f'/gts/{scene["name"]}/' + x + '/labels.npz' for x in all_token_with_order]
                self.all_samples.extend(path)
                init_pos += len(path)

                info_seq = self.infos[scene["name"]]
                for token, sample in zip(all_token_with_order, info_seq):
                    self.traj.append(sample['gt_ego_fut_trajs'][0])
            # TODO: make it support non-cache cfm training
            self.select_valid(training, vae_training=True)

        else:
            assert dataset_cfg.get('pickle_path', None) is not None, "Should provide cached pickles"

            path = dataset_cfg['pickle_path']['train' if training else 'test']
            with open(path, 'rb') as f:
                cached_files = pickle.load(f)
            gt_path = [x['gt_path'][0] for x in cached_files]
            token_seq = [x[0].split('/')[4] if isinstance(x, list) else x.split('/')[4] for x in gt_path]
            token_ori_sort = []
            for scene_idx, value in self.infos.items():
                for frame in value:
                    token_ori_sort.append(frame['token'])
            indices = [token_seq.index(token) for token in token_ori_sort if token in token_seq]
            sorted_cache_file = [cached_files[x] for x in indices]

            self.traj = [x['gt_trajs'] for x in sorted_cache_file]
            self.all_samples = [x['gt_path'][0] for x in sorted_cache_file]
            self.x_sampled = [x['x_sampled'] for x in sorted_cache_file]
            self.select_valid(training) # cache mode only during cfm training

    def select_valid(self, training, vae_training=False):
        self.valid_idx = []
        self.safe_length = self.sequence_length * 2 if not vae_training else self.sequence_length
        scenes_list = [x[0].split('/')[3] if isinstance(x, list) else x.split('/')[3] for x in self.all_samples]
        for idx, scene in enumerate(scenes_list):
            sub_seq = scenes_list[idx: idx + self.safe_length]
            if len(set(sub_seq)) == 1 and len(sub_seq) == self.safe_length:
                self.valid_idx.append(idx)
        self.valid_idx = self.valid_idx[::self.win_size] if training else self.valid_idx

    def __getitem__(self, idx):

        sample_idx = self.valid_idx[idx]
        data_dict = {
            'paths': self.all_samples[sample_idx: sample_idx + self.safe_length],
            'trajectory': np.concat(self.traj[sample_idx: sample_idx + self.safe_length])
        }
        if self.cache_mode:
            x_sampled = np.concat(self.x_sampled[sample_idx: sample_idx + self.safe_length])

            repeat = self.sequence_length - self.hist_last
            if repeat > 0:
                x_sampled[:repeat] = 0
                data_dict['trajectory'][:repeat] = 0
            data_dict['x_sampled'] = x_sampled
        else:
            paths = data_dict['paths']
            data_dict['semantic_occ'] = [np.load(path)['semantics'] for path in paths] if len(paths) > 1 else np.load(paths[0])['semantics']
        return data_dict
