from pathlib import Path
import hashlib,json,socket

def ensure(ok,message):
 if not ok: raise ValueError(message)

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(t,n=None):
 import torch
 t=t.detach().reshape(-1)
 n=t.numel() if n is None else int(n)
 ensure(n>0 and t.numel()>=n,'Restored tensor partition is too short')
 v=torch.cat((t[:min(256,n)].cpu(),t[max(0,n-256):n].cpu())).contiguous()
 ensure(bool(torch.isfinite(v).all()),'Non-finite restored parameter or optimizer moment')
 return {'numel':n,'dtype':str(t.dtype),'sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()}

def validate_recovery(cfg,*,smoke=False):
 ensure(not smoke,'This entry continues an already verified run; no fresh smoke or optimizer reset')
 p=Path(str(cfg['continuation_manifest_path'])).resolve(strict=True)
 m=json.loads(p.read_text())
 ensure(m['status']=='CPU_CHECKPOINT_STRUCTURAL_PASS' and m['source_job_id']=='dlcy3damjh1g5s7f','Unverified source experiment')
 ensure(Path(str(cfg['resume'])).resolve(strict=True)==Path(m['checkpoint_dir']).resolve(strict=True),'Resume must use the verified full state directory')
 ensure(digest(m['source_config'])==m['source_config_sha256'],'Original experiment configuration changed')
 ensure(digest(Path(m['checkpoint_dir'])/'trainer_state.json')==m['trainer_state_sha256'],'Saved progress changed')
 c=cfg['wholebody_contract']
 ensure(int(c['world_size'])==m['world_size']==32 and int(cfg['batch_size'])==m['microbatch']==8,'Resume distribution or microbatch changed')
 ensure(int(cfg['max_steps'])==int(cfg['lr_scheduler_total_steps'])==m['total_steps']==7920 and int(cfg['num_epochs'])==m['epochs']==60,'Resume must finish the original total60 epochs')
 ensure(c['statistics_sha256']==m['statistics_sha256'],'Normalization differs from original experiment')
 ensure(0<m['trainer_state']['global_step']<7920,'Checkpoint is outside the original schedule')
 for rel,size in m['files'].items():
  q=Path(m['checkpoint_dir'])/rel
  ensure(q.is_file() and q.stat().st_size==size,'Checkpoint missing, incomplete or changed: '+rel)
 return m

def verify_recovery(trainer,m):
 import torch
 import torch.distributed as dist
 a=trainer.accelerator
 engine=a.deepspeed_engine_wrapped.engine
 progress=m['trainer_state']
 ensure(trainer.global_step==engine.global_steps==progress['global_step'] and engine.skipped_steps==0,'Native optimizer or trainer progress was not restored')
 ensure(trainer.epoch==progress['epoch'] and trainer.batch_in_epoch==progress['batch_in_epoch'],'Dataloader epoch/batch position was not restored')
 native=getattr(trainer.scheduler,'scheduler',trainer.scheduler)
 ensure(native.last_epoch==m['scheduler_last_epoch'],'Learning-rate scheduler counter was reset')
 ensure(abs(float(trainer.optimizer.param_groups[0]['lr'])-m['scheduler_lr'])<1e-12,'Optimizer LR was reset')
 model=a.unwrap_model(trainer.model)
 state=model.state_dict()
 for key,expected in m['model_fingerprints'].items():
  ensure(key in state and fingerprint(state[key],expected['numel'])==expected,'Restored model tensor differs from saved experiment: '+key)
 saved=m['optimizer_partitions'][str(a.process_index)]
 masters=engine.optimizer.single_partition_of_fp32_groups
 ensure(len(masters)==len(saved['masters']),'FP32 optimizer partition group count differs')
 for value,expected in zip(masters,saved['masters']):
  ensure(fingerprint(value,expected['numel'])==expected,'FP32 master weights were not restored correctly')
 base=engine.optimizer.optimizer
 ensure(len(base.param_groups)==len(saved['moments']),'Optimizer state groups differ')
 for group,expected in zip(base.param_groups,saved['moments']):
  ensure(len(group['params'])==1,'Unexpected flattened optimizer layout')
  st=base.state[group['params'][0]]
  step=int(st['step'].item()) if torch.is_tensor(st['step']) else int(st['step'])
  ensure(step==expected['step'],'Adam step counter was reset')
  for key in ['exp_avg','exp_avg_sq']:
   ensure(fingerprint(st[key],expected[key]['numel'])==expected[key],'Adam moment was not restored: '+key)
 record={'status':'FULL_STATE_RESTORE_VERIFIED','rank':a.process_index,'local_rank':a.local_process_index,'hostname':socket.gethostname(),'world_size':a.num_processes,'checkpoint_dir':m['checkpoint_dir'],'global_step':trainer.global_step,'epoch':trainer.epoch,'batch_in_epoch':trainer.batch_in_epoch,'learning_rate':float(trainer.optimizer.param_groups[0]['lr']),'model_samples_verified':True,'fp32_master_samples_verified':True,'adam_moment_samples_verified':True,'original_total_steps':7920,'original_total_epochs':60}
 records=[None]*a.num_processes
 dist.all_gather_object(records,record)
 ensure(len({(x['hostname'],x['local_rank']) for x in records})==32 and len({x['hostname'] for x in records})==4,'Expected4 nodes and32 different GPU ranks')
 ensure(all(x['global_step']==progress['global_step'] and x['status']=='FULL_STATE_RESTORE_VERIFIED' for x in records),'Not all ranks restored the same step')
 p=Path(trainer.output_dir)/f'restore_rank_{a.process_index}.json'
 with p.open('x') as f:json.dump(record,f,indent=2)
 a.wait_for_everyone()
 if a.is_main_process: print('FULL_STATE_RESTORE_VERIFIED all32 ranks step='+str(progress['global_step'])+'; continuing original60 epochs',flush=True)
