import argparse
import json
from pathlib import Path

from imagewam import runtime
from imagewam.trainer import Wan22Trainer
from omegaconf import OmegaConf


class H2Trainer(Wan22Trainer):
    def _resume_or_load_checkpoint(self):
        if not self.resume or len(self.train_dataset)!=72752 or len(self.train_loader)!=285 or self.accelerator.num_processes!=32:
            raise RuntimeError('Unexpected H2 training world, collection or resume state')
        result=super()._resume_or_load_checkpoint()
        engine=self.accelerator.deepspeed_engine_wrapped.engine
        if not (0 < self.global_step < self.max_steps) or engine.global_steps!=self.global_step:
            raise RuntimeError('H2 recovery did not restore the optimizer step')
        rank=self.accelerator.process_index
        (Path(self.output_dir)/f'ready-rank-{rank:02d}.json').write_text(json.dumps(dict(rank=rank,step=self.global_step,epoch_steps=len(self.train_loader),frames=len(self.train_dataset)))+'\n')
        return result

    def train(self):
        super().train()
        engine=self.accelerator.deepspeed_engine_wrapped.engine
        if self.global_step!=self.max_steps or engine.global_steps!=self.global_step or engine.skipped_steps!=0:
            raise RuntimeError('H2 training exited before its target step')
        rank=self.accelerator.process_index
        (Path(self.output_dir)/f'finish-rank-{rank:02d}.json').write_text(json.dumps(dict(rank=rank,global_step=self.global_step,optimizer_steps=engine.global_steps,skipped_steps=engine.skipped_steps,epoch=self.epoch,batch_in_epoch=self.batch_in_epoch))+'\n')
        self.accelerator.wait_for_everyone()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    cfg=OmegaConf.load(args.config)
    if (cfg.model.expected_action_dim,cfg.model.action_state_dim)!=(75,78) or not cfg.resume or cfg.model.get('parent_checkpoint'):
        raise RuntimeError('Expected new H2 action head and no prior task checkpoint')
    if not args.smoke:
        root=Path(__file__).resolve().parents[3]/'run'
        receipts=[json.loads(x.read_text()) for x in (root/'smoke').glob('finish-rank-*.json')]
        if {r['rank'] for r in receipts}!=set(range(32)) or any(r['global_step']!=1 or r['optimizer_steps']!=1 or r['skipped_steps']!=0 for r in receipts):
            raise RuntimeError('Successful 32-rank smoke is required before production')
    runtime.Wan22Trainer=H2Trainer
    runtime.run_training(cfg)
