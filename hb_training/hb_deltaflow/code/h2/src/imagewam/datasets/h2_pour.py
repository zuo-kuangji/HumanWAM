from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .wholebody import DEFAULT_PROMPT, WholeBodyLeRobotDataset, decode_resized_frames, window_indices
from .wholebody_augmented import augment_views_rgb01, sample_parameters

CAMERAS = ('observation.images.head', 'observation.images.chest')
SCHEMA = 'a18-h2-sharpa-body31-hand44-state78-chunk16-v1'
ACTION_DIM = 75
STATE_DIM = 78


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_collection(root):
    info = json.loads((root/'meta/info.json').read_text())
    episodes = sorted(read_jsonl(root/'meta/episodes.jsonl'), key=lambda row: row['episode_index'])
    tasks = read_jsonl(root/'meta/tasks.jsonl')
    if info['robot_type']!='h2_sharpa' or info['fps']!=30 or len(episodes)!=100 or len(tasks)!=1:
        raise ValueError('Unexpected A18 H2 collection')
    if [row['episode_index'] for row in episodes]!=list(range(100)) or sum(row['length'] for row in episodes)!=72752:
        raise ValueError('Invalid A18 episode IDs or total frame count')
    f=info['features']
    names=f['observation.state']['names']
    if (len(names)!=75 or len(set(names))!=75 or names[:31]!=f['action.body_target']['names']
            or names[31:]!=f['action.hands']['names'] or names[15:29]!=f['action.arms']['names']):
        raise ValueError('H2 joint order mismatch')
    return episodes,tasks[0]['task'],names


def load_episode(root, episode, n):
    fields=('observation.state','observation.projected_gravity','action.body_target','action.hands','action.arms')
    tab=pq.read_table(root/f'data/chunk-000/episode_{episode:06d}.parquet',columns=list(fields))
    q,g,b,h,arms=(np.asarray(tab[field].to_pylist(),dtype=np.float32) for field in fields)
    if [x.shape for x in (q,g,b,h,arms)]!=[(n,75),(n,3),(n,31),(n,44),(n,14)]:
        raise ValueError(f'Invalid H2 vector shapes in episode {episode}')
    if not np.allclose(b[:,15:29],arms,atol=1e-5,rtol=0):
        raise ValueError(f'Arm overrides do not match body targets in episode {episode}')
    state=np.concatenate([q,g],axis=1)
    action=np.concatenate([b,h],axis=1)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f'Non-finite H2 features in episode {episode}')
    return state,action


def fit(root, path):
    episodes,instruction,names=load_collection(root)
    pairs=[load_episode(root,row['episode_index'],row['length']) for row in episodes]
    stats=dict(schema=SCHEMA,instruction=instruction,episode_ids=list(range(100)),frames=72752,joint_names=names,cameras=list(CAMERAS))
    for kind,data in zip(('state','action'),zip(*pairs)):
        lo,hi=np.quantile(np.concatenate(data),[.01,.99],axis=0)
        stats[kind]=dict(q01=lo.tolist(),q99=hi.tolist())
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(stats,indent=2)+'\n')
    tmp.replace(path)


class H2PourDataset(torch.utils.data.Dataset):
    _frames=WholeBodyLeRobotDataset._frames

    def __init__(self,dataset_dir,stats_path,qwen_text_cache_dir,qwen_context_len,qwen_text_dim,video_cache_frames=64,image_augmentation=None):
        self.root=Path(dataset_dir).resolve(strict=True)
        episodes,task,joints=load_collection(self.root)
        stats=json.loads(Path(stats_path).read_text())
        if (stats['schema'],stats['instruction'],stats['episode_ids'],stats['joint_names'],stats['cameras'])!=(SCHEMA,task,list(range(100)),joints,list(CAMERAS)):
            raise ValueError('Normalization does not match the H2 task')
        self.transforms={}
        for name,dim in [('state',STATE_DIM),('action',ACTION_DIM)]:
            lo=np.asarray(stats[name]['q01'],dtype=np.float32)
            hi=np.asarray(stats[name]['q99'],dtype=np.float32)
            if lo.shape!=(dim,) or hi.shape!=(dim,) or not np.isfinite([lo,hi]).all() or (hi<lo).any():
                raise ValueError(f'Bad H2 normalization bounds for {name}')
            span=hi-lo
            constant=span<1e-4
            span[constant]=2
            scale=2/span
            offset=-1-scale*lo
            offset[constant]=-lo[constant]
            self.transforms[name]=(scale,offset)
        self.episode_ids=[e['episode_index'] for e in episodes]
        self.ends=np.cumsum([e['length'] for e in episodes])
        self.episodes={e['episode_index']:load_episode(self.root,e['episode_index'],e['length']) for e in episodes}
        self.prompt=DEFAULT_PROMPT.format(task=task)
        key=hashlib.sha256(self.prompt.encode()).hexdigest()
        cache=Path(qwen_text_cache_dir)/f'{key}.qwen3_flux2_len{qwen_context_len}.pt'
        payload=torch.load(cache,map_location='cpu',weights_only=True)
        self.text_hidden_states=payload['text_hidden_states']
        self.text_attention_mask=payload['text_attention_mask'].bool()
        if (self.text_hidden_states.shape!=(qwen_context_len,qwen_text_dim) or self.text_attention_mask.shape!=(qwen_context_len,)
                or not torch.isfinite(self.text_hidden_states).all() or not self.text_attention_mask.any()):
            raise ValueError('Invalid A18 instruction cache')
        self.video_cache_frames=int(video_cache_frames)
        self.frame_cache=OrderedDict()
        self.decoder=decode_resized_frames
        self.image_augmentation=dict(image_augmentation or {})
        self.rng=None

    def __len__(self):
        return int(self.ends[-1])

    def normalize(self,value,kind):
        scale,offset=self.transforms[kind]
        return np.clip(value*scale+offset,-5,5).astype(np.float32)

    def __getitem__(self,index):
        ep_position=int(np.searchsorted(self.ends,index,side='right'))
        episode=self.episode_ids[ep_position]
        anchor=int(index-(self.ends[ep_position-1] if ep_position else 0))
        state,action=self.episodes[episode]
        window=window_indices(anchor,len(action),16,16)
        views=torch.stack([self._frames(episode,c,window['image'].tolist()) for c in CAMERAS],dim=1)
        if self.image_augmentation.get('enabled',False):
            if self.rng is None:
                self.rng=np.random.default_rng(torch.initial_seed())
            params=sample_parameters(self.rng,self.image_augmentation)
            views=augment_views_rgb01(views,params,noise_seed=int(self.rng.integers(0,2**63-1)),input_frame_mask=torch.tensor([True,False]))
        video=(torch.cat(list(views.unbind(1)),dim=-1)*2-1).permute(1,0,2,3).contiguous()
        future=self.normalize(action[window['action']],'action')
        past=self.normalize(action[window['past_action']],'action')
        if window['past_action_is_pad'].all():
            past=np.zeros_like(past)
        current=self.normalize(state[anchor],'state')
        return dict(video=video,action=torch.from_numpy(future),past_action=torch.from_numpy(past),proprio=torch.from_numpy(np.repeat(current[None],16,axis=0)),action_is_pad=torch.from_numpy(window['action_is_pad']),past_action_is_pad=torch.from_numpy(window['past_action_is_pad']),image_is_pad=torch.from_numpy(window['image_is_pad']),proprio_is_pad=torch.zeros(16,dtype=torch.bool),action_dim_is_pad=torch.zeros(ACTION_DIM,dtype=torch.bool),proprio_dim_is_pad=torch.zeros(STATE_DIM,dtype=torch.bool),text_hidden_states=self.text_hidden_states,text_attention_mask=self.text_attention_mask,prompt=self.prompt,instruction=self.prompt,idx=int(index),episode_index=episode,frame_index=anchor,task_index=0,dataset_index=0)
