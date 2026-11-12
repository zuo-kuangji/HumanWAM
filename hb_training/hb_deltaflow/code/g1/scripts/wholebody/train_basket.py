import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from imagewam import runtime
from imagewam.trainer import Wan22Trainer


class BasketTrainer(Wan22Trainer):
    def _resume_or_load_checkpoint(self):
        if len(self.train_dataset) != 54337 or len(self.train_loader) != 213 or self.accelerator.num_processes != 32:
            raise RuntimeError('Expected the pinned 100-episode G1 basket collection and 32 GPUs')
        return super()._resume_or_load_checkpoint()

    def save_checkpoint(self):
        if self.global_step == self.max_steps:
            return super().save_checkpoint()
        self.accelerator.wait_for_everyone()
        weights = None
        if self.accelerator.is_main_process:
            weights = self._save_weights_checkpoint(step_tag=f'step_{self.global_step:06d}')
            self._prune_old_weight_checkpoints()
        self.accelerator.wait_for_everyone()
        return {'weights_path': weights, 'state_path': None}

    def train(self):
        super().train()
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        if self.global_step != self.max_steps or engine.skipped_steps != 0:
            raise RuntimeError('Training ended without reaching its optimizer-update target')
        if not self.resume and engine.global_steps != self.global_step:
            raise RuntimeError('Trainer and optimizer update counts differ')
        rank = self.accelerator.process_index
        path = Path(self.output_dir) / f'finish-rank-{rank:02d}.json'
        path.write_text(json.dumps({'rank': rank, 'global_step': self.global_step, 'optimizer_steps': engine.global_steps}) + '\n')
        self.accelerator.wait_for_everyone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    folder = args.config.parent
    root = folder.parent
    if not (root / 'data-verified.json').is_file():
        raise RuntimeError('Pinned dataset verification is required')
    if not args.smoke:
        receipts = [json.loads(path.read_text()) for path in (folder / 'smoke').glob('finish-rank-*.json')]
        if len(receipts) != 32 or {item['rank'] for item in receipts} != set(range(32)):
            raise RuntimeError('Run the distributed short check first')
        if any(item['global_step'] != 1 or item['optimizer_steps'] != 1 for item in receipts):
            raise RuntimeError('The distributed short check did not complete its optimizer update')
        if any((folder / 'node-logs' / f'smoke-node-{rank}.exit').read_text().strip() != '0' for rank in range(4)):
            raise RuntimeError('A short-check launcher did not exit cleanly')
    runtime.Wan22Trainer = BasketTrainer
    runtime.run_training(cfg)
