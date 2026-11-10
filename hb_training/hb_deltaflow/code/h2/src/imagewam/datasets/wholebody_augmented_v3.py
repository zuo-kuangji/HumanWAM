"""B6 v3: input-only proprioception/latency and coherent visual randomization."""
from __future__ import annotations
import io, math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from .wholebody import WholeBodyLeRobotDataset
from .wholebody_augmented import WholeBodyAugmentedLeRobotDataset, sample_parameters, augment_views_rgb01

class WholeBodyRobustLeRobotDataset(WholeBodyAugmentedLeRobotDataset):
    def __init__(self,*args,proprio_augmentation=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.proprio_augmentation=dict(proprio_augmentation or {})
        self.last_augmentation={}
    def augment_video(self,video,rng):
        x=(video.permute(1,0,2,3)+1.0).mul(0.5).clamp(0,1)
        views=torch.stack(x.chunk(2,dim=-1),dim=1)
        policy=self.image_augmentation
        angles=[]
        for v in range(views.shape[1]):
            angle=float(rng.uniform(-policy['rotation_degrees'],policy['rotation_degrees']))
            scale=1.0+abs(angle)*0.015+float(rng.uniform(0,policy['crop_zoom_extra']))
            views[:,v]=TF.affine(views[:,v],angle=angle,translate=[0,0],scale=scale,shear=[0,0],interpolation=InterpolationMode.BILINEAR)
            angles.append(angle)
        params=sample_parameters(rng,policy)
        mask=torch.ones(views.shape[0],dtype=torch.bool);mask[-1]=False
        result=augment_views_rgb01(views,params,noise_seed=int(rng.integers(0,2**63-1)),input_frame_mask=mask)
        jpeg_quality=None
        if rng.random()<policy['jpeg_probability']:
            jpeg_quality=int(rng.integers(policy['jpeg_quality'][0],policy['jpeg_quality'][1]+1))
            for t in range(result.shape[0]-1):
                for v in range(result.shape[1]):
                    buffer=io.BytesIO()
                    TF.to_pil_image(result[t,v]).save(buffer,format='JPEG',quality=jpeg_quality)
                    buffer.seek(0)
                    with Image.open(buffer) as image:
                        result[t,v]=TF.pil_to_tensor(image.convert('RGB')).to(dtype=result.dtype).div(255.0)
        joined=torch.cat(list(result.unbind(1)),dim=-1)
        return (joined*2.0-1.0).permute(1,0,2,3).contiguous(),dict(params=params,angles=angles,jpeg_quality=jpeg_quality)
    def augment_numeric(self,sample,rng):
        policy=self.proprio_augmentation
        episode=int(sample['episode_index']);anchor=int(sample['frame_index'])
        delay=0
        if anchor>0 and rng.random()<policy['delay_probability']:
            cycles=int(rng.integers(1,policy['max_delay_control_cycles']+1))
            delay=max(1,round(cycles*policy['recording_hz']/policy['control_hz']))
        states,_=self.episodes[episode]
        state=states[max(0,anchor-delay)].copy()
        bias_rng=np.random.default_rng(np.random.SeedSequence([policy['seed'],episode,3719]))
        bias=bias_rng.uniform(-math.radians(policy['joint_bias_degrees']),math.radians(policy['joint_bias_degrees']),29)
        sigma=math.radians(policy['joint_noise_std_degrees'])
        noise=np.clip(rng.normal(0,sigma,29),-3*sigma,3*sigma)
        state[:29]=(state[:29]+bias+noise).astype(np.float32)
        current=self.normalizer.forward(state,'state')
        sample['proprio']=torch.from_numpy(np.repeat(current[None],self.action_horizon,axis=0)).to(dtype=sample['proprio'].dtype)
        past=sample['past_action'].clone();pad=sample['past_action_is_pad'].clone()
        if delay:
            shifted=torch.zeros_like(past);shift_pad=torch.ones_like(pad)
            if delay<len(past):
                shifted[delay:]=past[:-delay];shift_pad[delay:]=pad[:-delay]
            past,pad=shifted,shift_pad
        dropped=[]
        if rng.random()<policy['history_drop_probability']:
            candidates=[i for i in range(1,len(past)) if not bool(pad[i-1]) and not bool(pad[i])]
            if candidates:
                i=int(rng.choice(candidates));past[i]=past[i-1];pad[i]=pad[i-1];dropped.append(i)
        past[pad]=0
        sample['past_action']=past;sample['past_action_is_pad']=pad
        return dict(delay_recorded_frames=delay,bias_degrees=(bias*180/math.pi).tolist(),noise_degrees=(noise*180/math.pi).tolist(),history_drop_rows=dropped)
    def __getitem__(self,index):
        sample=WholeBodyLeRobotDataset.__getitem__(self,index)
        rng=self._rng()
        sample['video'],visual=self.augment_video(sample['video'],rng)
        numeric=self.augment_numeric(sample,rng)
        self.last_augmentation=dict(visual=visual,numeric=numeric)
        return sample
