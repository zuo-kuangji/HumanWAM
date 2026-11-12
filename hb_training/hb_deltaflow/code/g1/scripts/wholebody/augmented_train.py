"""Approved B6 online augmentation fine-tune: parent model only, fresh optimizer."""
from pathlib import Path
import sys,os,json,math,inspect
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.wholebody import train as base
from omegaconf import OmegaConf
PHASE=Path('/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/augmentation-review-20261105/finetune-v2-20261105')
PARENT=Path('/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/continuation-20261105-30d').parent.parent/'checkpoint-exports/b6-v1v2-epoch40-step5280/model_step_005280.pt'
MODEL_TARGET='imagewam.wholebody_delta.create_imagewam_flux2_wholebody_delta'
DATA_TARGET='imagewam.datasets.wholebody_augmented.WholeBodyAugmentedLeRobotDataset'

def validate(cfg,*,smoke):
 c=cfg.wholebody_contract;ds=cfg.data.train
 base.require(cfg.model._target_==MODEL_TARGET and ds._target_==DATA_TARGET,'Wrong model or augmentation loader')
 base.require(Path(str(cfg.resume)).resolve()==PARENT.resolve() and PARENT.is_file(),'Must load preserved epoch40 model file only')
 base.require(cfg.data.val is None and cfg.eval_every==0 and ds.train_split_mode=='all_60','All60 dataset contract changed')
 base.require(ds.action_horizon==ds.past_action_size==16 and cfg.model.action_state_dim==44 and cfg.model.expected_action_dim==76,'Protocol dimensions changed')
 base.require(len(ds.episode_ids)==60 if 'episode_ids' in ds else True,'Episode count changed')
 base.require(cfg.num_epochs==20 and cfg.batch_size==8 and cfg.gradient_accumulation_steps==1 and c.world_size==32,'20epoch/32rank batch contract changed')
 base.require(float(cfg.learning_rate)==2e-5 and cfg.weight_decay==0.01 and cfg.mixed_precision=='bf16','Fine-tune optimizer recipe changed')
 base.require(cfg.lr_scheduler_total_steps==2640 and cfg.warmup_steps==132 and c.updates_per_epoch==132,'Fine-tune schedule differs')
 base.require(cfg.max_steps==(5 if smoke else 2640),'Stopping count differs')
 base.require(base.sha256(Path(ds.stats_path))==c.statistics_sha256,'Normalization statistics changed')
 wanted=json.loads((PHASE/'approved_policy.json').read_text())
 base.require(OmegaConf.to_container(ds.image_augmentation,resolve=True)==wanted,'Augmentation policy differs from approved v2')
 base.require(bool(wanted['enabled']) and wanted['saturation']==[0.76,1.24] and wanted['contrast']==[0.88,1.12],'Approved strengths missing')
 if smoke:base.require(cfg.save_every==0 and not cfg.save_at_end and not cfg.save_steps,'Preflight must save no model/optimizer state')
 else:base.require(cfg.save_every==1320 and cfg.save_at_end and cfg.keep_last_weights==4 and cfg.keep_latest_state_only,'Production checkpoint policy differs')
 receipt=json.loads((PHASE/'run/build_receipt.json').read_text())
 for rel,digest in receipt['code_sha256'].items():base.require(base.sha256(Path(__file__).resolve().parents[2]/rel)==digest,'Code changed after preparation: '+rel)
 return {'updates_per_epoch':132,'total_updates':2640}

def require_smoke(cfg):
 for rank in range(32):
  p=PHASE/f'run/smoke/smoke_rank_{rank}.json';base.require(p.is_file(),'Missing five-step preflight rank '+str(rank));r=json.loads(p.read_text())
  base.require(r['status']=='FIVE_UPDATES_VERIFIED' and r['actual_optimizer_updates']==5 and r['skipped_updates']==0 and r['world_size']==32,'Five-step preflight failed')
  a=json.loads((PHASE/f'run/smoke/fine_tune_rank_{rank}_start.json').read_text())
  base.require(a['parent_model_loaded'] and a['optimizer_fresh'] and a['config_sha256']==base.sha256(PHASE/'run/smoke.yaml'),'Preflight parent or config mismatch')
 base.require(not any((PHASE/'run/smoke/checkpoints').rglob('*.pt')),'Preflight wrote a checkpoint')

def install(*,smoke):
 import torch
 from imagewam.wholebody_delta import StateOnlyDeltaImageWAM
 from imagewam.trainer import Wan22Trainer
 old_load=StateOnlyDeltaImageWAM.load_checkpoint
 def checked_load(model,path,optimizer=None):
  base.require(Path(path).resolve()==PARENT.resolve() and optimizer is None,'Only approved parent model weights may be loaded')
  payload=old_load(model,path,optimizer=None);saved=payload['mot'];actual=model.mot.state_dict();keys=list(saved)
  probes=list(dict.fromkeys([keys[0],keys[len(keys)//2],keys[-1]]+[k for k in keys if 'state_encoder' in k]))
  for key in probes:
   value=actual[key].detach().flatten()[:64].cpu();target=saved[key].detach().flatten()[:64].to(dtype=value.dtype,device='cpu');base.require(torch.equal(value,target),'Parent model probe mismatch: '+key)
  model._fine_tune_parent_verified={'tensor_count':len(keys),'probed_tensors':len(probes),'path':str(PARENT)}
  return payload
 StateOnlyDeltaImageWAM.load_checkpoint=checked_load
 old_init=Wan22Trainer.__init__;old_train=Wan22Trainer.train
 def checked_init(self,*args,**kwargs):
  old_init(self,*args,**kwargs);cfg=self.cfg;acc=self.accelerator;model=acc.unwrap_model(self.model);engine=acc.deepspeed_engine_wrapped.engine
  base.require(acc.num_processes==32 and engine.global_steps==0 and engine.skipped_steps==0 and self.global_step==0 and self.epoch==0,'Unexpected initial distributed progress')
  base.require(self._weights_resume_loaded_before_prepare and hasattr(model,'_fine_tune_parent_verified'),'Parent model not loaded before optimizer initialization')
  base.require(len(self.train_loader)==132 and self.max_steps==(5 if smoke else 2640),'Actual dataset/schedule differs')
  base.require(acc.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage']==1,'Expected ZeRO1')
  state=engine.optimizer.optimizer.state;steps=[]
  for value in state.values():
   if 'step' in value:steps.append(float(value['step']))
  base.require(all(x==0 for x in steps),'Optimizer contains old update steps')
  self._fine_tune_start={'status':'FRESH_OPTIMIZER_WITH_EPOCH40_MODEL','rank':acc.process_index,'world_size':32,'parent_model_loaded':True,'optimizer_fresh':True,'parent_model':model._fine_tune_parent_verified,'initial_global_step':self.global_step,'initial_optimizer_steps':steps,'config_sha256':base.sha256(PHASE/('run/smoke.yaml' if smoke else 'run/production.yaml')),'learning_rate':float(cfg.learning_rate),'schedule_total_steps':2640,'augmentation_policy_sha256':base.sha256(PHASE/'approved_policy.json')}
  base.write_json_new(Path(self.output_dir)/f'fine_tune_rank_{acc.process_index}_start.json',self._fine_tune_start)
 Wan22Trainer.__init__=checked_init
 def checked_train(self):
  old_train(self);engine=self.accelerator.deepspeed_engine_wrapped.engine
  base.require(self.global_step==self.max_steps==engine.global_steps and engine.skipped_steps==0,'Fine-tune failed exact update count')
  base.write_json_new(Path(self.output_dir)/f'fine_tune_rank_{self.accelerator.process_index}_complete.json',{'status':'FIVE_STEPS_COMPLETE' if smoke else 'TWENTY_AUGMENTED_EPOCHS_COMPLETE','rank':self.accelerator.process_index,'global_step':self.global_step,'skipped_updates':engine.skipped_steps,'parent_epoch':40,'new_epochs':20,'policy':'b6-visual-aug-v2'})
 Wan22Trainer.train=checked_train

base.validate_config=validate
base.require_completed_smoke=require_smoke
base.install_training_contract=install
source=inspect.getsource(base.install_smoke_hooks).replace('production_must_reload_fresh_base=True','production_must_reload_parent_weights=True')
exec(compile(source,__file__+':verified_smoke_hooks','exec'),base.__dict__)
if __name__=='__main__':base.main()
