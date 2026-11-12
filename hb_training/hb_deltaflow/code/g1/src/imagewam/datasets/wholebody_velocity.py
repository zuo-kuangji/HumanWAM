import numpy as np
import torch

from .wholebody_augmented_v3 import WholeBodyRobustLeRobotDataset


class WholeBodyVelocityLeRobotDataset(WholeBodyRobustLeRobotDataset):
    def __init__(self, *, velocity_cache, velocity_noise_std=0.05, **kwargs):
        super().__init__(**kwargs)
        self.velocity_noise_std = float(velocity_noise_std)
        with np.load(velocity_cache) as cache:
            self.velocity_mean = cache['mean']
            self.velocity_scale = cache['scale']
            self.velocities = {int(key.split('_')[1]): cache[key]
                               for key in cache.files if key.startswith('episode_')}
        for episode, (states, _) in self.episodes.items():
            if self.velocities[episode].shape != (len(states), 29):
                raise ValueError(f'Invalid velocity shape for episode {episode}')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        delay = self.last_augmentation['numeric']['delay_recorded_frames']
        frame = max(0, sample['frame_index'] - delay)
        if frame > 0:
            velocity = self.velocities[sample['episode_index']][frame]
            sigma = self.velocity_noise_std
            noise = self._rng().normal(0, sigma, size=velocity.shape).clip(-3 * sigma, 3 * sigma)
            normalized = np.clip((velocity + noise - self.velocity_mean) / self.velocity_scale, -5, 5)
        else:
            normalized = np.zeros(29, dtype=np.float32)
        proprio = sample['proprio']
        channel = torch.as_tensor(normalized, dtype=proprio.dtype).expand(proprio.shape[0], -1)
        sample['proprio'] = torch.cat((proprio, channel), dim=-1)
        sample['proprio_dim_is_pad'] = torch.zeros(73, dtype=torch.bool)
        return sample
