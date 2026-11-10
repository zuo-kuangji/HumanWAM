"""Sequence-consistent B6 visual augmentation; action/state supervision is untouched."""
from __future__ import annotations
import os
from typing import Any
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .wholebody import WholeBodyLeRobotDataset

DEFAULT_POLICY={"seed":20261105,"enabled":True,"exposure_ev":[-0.2,0.2],"contrast":[0.9,1.1],"saturation":[0.8,1.2],"hue":[-0.015,0.015],"gamma":[0.9,1.1],"temperature_delta":[-0.06,0.06],"noise_probability":0.35,"noise_sigma":[0.003,0.012],"blur_probability":0.15,"blur_sigma":[0.25,0.65],"blur_kernel":3}

def sample_parameters(rng: np.random.Generator, policy: dict[str,Any]|None=None)->dict[str,float]:
 p={**DEFAULT_POLICY,**(policy or {})}
 if not p.get("enabled",True):raise ValueError("This phase requires visual augmentation on every sample")
 result={k:float(rng.uniform(*p[k])) for k in ("exposure_ev","contrast","saturation","hue","gamma","temperature_delta")}
 result["noise_sigma"]=float(rng.uniform(*p["noise_sigma"])) if rng.random()<p["noise_probability"] else 0.0
 result["blur_sigma"]=float(rng.uniform(*p["blur_sigma"])) if rng.random()<p["blur_probability"] else 0.0
 return result

def augment_views_rgb01(rgb: torch.Tensor, params: dict[str,float], *, noise_seed:int, input_frame_mask:torch.Tensor|None=None)->torch.Tensor:
 """Transform [time,view,3,H,W] in RGB[0,1], sharing colour/optics across all frames/views."""
 if rgb.ndim!=5 or rgb.shape[2]!=3 or not rgb.is_floating_point():raise ValueError("Expected float [T,V,3,H,W]")
 t,v,c,h,w=rgb.shape
 x=rgb.reshape(t*v,c,h,w).clone().clamp(0,1)
 x=(x*float(2.0**params["exposure_ev"])).clamp(0,1)
 delta=params["temperature_delta"]
 gains=x.new_tensor([1.0+delta,1.0,1.0-delta]).view(1,3,1,1)
 x=(x*gains).clamp(0,1)
 x=((x-0.5)*params["contrast"]+0.5).clamp(0,1)
 x=TF.adjust_saturation(x,params["saturation"])
 x=TF.adjust_hue(x,params["hue"])
 x=x.clamp(0,1).pow(params["gamma"])
 if params.get("blur_sigma",0)>0:
  x=TF.gaussian_blur(x,[3,3],[params["blur_sigma"]]*2)
 x=x.reshape(t,v,c,h,w)
 sigma=params.get("noise_sigma",0.0)
 if sigma>0:
  generator=torch.Generator(device=x.device).manual_seed(int(noise_seed))
  noise=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)*sigma
  mask=torch.ones(t,dtype=torch.bool,device=x.device) if input_frame_mask is None else input_frame_mask.to(x.device,dtype=torch.bool)
  if mask.shape!=(t,):raise ValueError("Noise mask must contain one flag per temporal image")
  x=x+noise*mask.view(t,1,1,1,1)
 return x.clamp(0,1)

class WholeBodyAugmentedLeRobotDataset(WholeBodyLeRobotDataset):
 """Same numeric labels, sample indices and normalization as the verified all60 loader."""
 def __init__(self,*args,image_augmentation:dict[str,Any]|None=None,**kwargs):
  super().__init__(*args,**kwargs)
  self.image_augmentation={**DEFAULT_POLICY,**(image_augmentation or {})}
  if not self.image_augmentation.get("enabled",True):raise ValueError("Augmentation must stay enabled")
  self._augmentation_rng=None;self._augmentation_worker_key=None
 def _rng(self):
  worker=torch.utils.data.get_worker_info();worker_id=worker.id if worker else 0
  rank=int(os.environ.get("RANK","0"));initial=int(torch.initial_seed())
  key=(os.getpid(),rank,worker_id,initial)
  if key!=self._augmentation_worker_key:
   self._augmentation_rng=np.random.default_rng(np.random.SeedSequence([int(self.image_augmentation["seed"]),rank,worker_id,initial%(2**32),initial//(2**32)]))
   self._augmentation_worker_key=key
  return self._augmentation_rng
 def __getitem__(self,index):
  sample=super().__getitem__(index)
  video=sample["video"]
  if video.ndim!=4 or video.shape[0]!=3 or video.shape[-1]%2:raise ValueError("Expected [3,T,H,2W] with ego/chest views")
  x=((video.permute(1,0,2,3)+1.0)*0.5).clamp(0,1)
  views=torch.stack(x.chunk(2,dim=-1),dim=1)
  rng=self._rng();params=sample_parameters(rng,self.image_augmentation)
  # The final temporal image is the supervised t+16 image: no random sensor noise on that target.
  mask=torch.ones(views.shape[0],dtype=torch.bool);mask[-1]=False
  output=augment_views_rgb01(views,params,noise_seed=int(rng.integers(0,2**63-1)),input_frame_mask=mask)
  joined=torch.cat(list(output.unbind(1)),dim=-1)
  sample["video"]=(joined*2.0-1.0).permute(1,0,2,3).contiguous()
  return sample
