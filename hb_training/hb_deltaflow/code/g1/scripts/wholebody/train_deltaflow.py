"""ImageWAM Delta-flow 40-epoch training entry (B6/A3/A7, 100 episodes, augmentation v2).

Installed by prepare_deltaflow.py as HumanWAM/scripts/wholebody/train_deltaflow.py. Derived from train_a3a7.py:
- same native runtime/trainer, same 32-GPU loader-length checks and finish receipts;
- no separate 1-update smoke phase (the supervisor retries / resumes instead);
- every save (epoch 10/20/30/40) writes weights + full training state; the trainer keeps all four weight files
  (keep_latest_weights=4) and only the newest training state (keep_latest_state_only=true);
- --resume <run/production/checkpoints/state/step_N> continues an interrupted run from its newest state.
"""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from imagewam import runtime
from imagewam.trainer import Wan22Trainer

DATASET_TARGET = 'imagewam.datasets.wholebody_hbaug.WholeBodyHBAugV2LeRobotDataset'
AUG_VERSION = 'hb-aug-v2-20261109'
EPOCHS, SAVE_EVERY_EPOCHS, WARMUP_EPOCHS, WORLD = 40, 10, 2, 32
EXPECTED = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class DeltaFlowTrainer(Wan22Trainer):
    def _resume_or_load_checkpoint(self):
        require(len(self.train_dataset) == EXPECTED['frames'] and len(self.train_loader) == EXPECTED['updates_per_epoch']
                and self.accelerator.num_processes == WORLD,
                'Expected the pinned 100-episode collection, its updates/epoch and 32 GPUs')
        require(type(self.train_dataset).__name__ == DATASET_TARGET.rsplit('.', 1)[1], 'Augmentation v2 dataset missing')
        return super()._resume_or_load_checkpoint()

    def train(self):
        start = int(self.global_step)
        super().train()
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        require(self.global_step == self.max_steps and engine.skipped_steps == 0,
                'Training ended without reaching its optimizer-update target')
        if not self.resume:
            require(engine.global_steps == self.global_step, 'Trainer and optimizer update counts differ')
        rank = self.accelerator.process_index
        path = Path(self.output_dir) / f'finish-rank-{rank:02d}.json'
        path.write_text(json.dumps({'rank': rank, 'global_step': self.global_step,
                                    'optimizer_steps': int(engine.global_steps), 'resumed_from_step': start}) + '\n')
        self.accelerator.wait_for_everyone()


def check_config(cfg, root, resume):
    receipt = json.loads((root / 'data-verified.json').read_text())
    u = int(receipt['updates_per_epoch'])
    total = EPOCHS * u
    EXPECTED.update(frames=int(receipt['frames']), updates_per_epoch=u)
    d = cfg.data.train
    require(d._target_ == DATASET_TARGET and d.image_augmentation.version == AUG_VERSION
            and d.image_augmentation.enabled is True, 'Config does not use augmentation v2')
    require(d.train_split_mode == 'first_100' and receipt['train_episode_ids'] == list(range(100)),
            'Training must use exactly episodes 0-99')
    require(cfg.num_epochs == EPOCHS and cfg.max_steps == total and cfg.lr_scheduler_total_steps == total
            and cfg.warmup_steps == WARMUP_EPOCHS * u, 'Epoch/step schedule differs from 40 x updates_per_epoch')
    require(list(cfg.save_steps) == [e * u for e in range(SAVE_EVERY_EPOCHS, EPOCHS + 1, SAVE_EVERY_EPOCHS)]
            and cfg.save_every == 0 and cfg.save_at_end is True and cfg.keep_latest_weights == 4
            and cfg.keep_latest_state_only is True, 'Checkpoint policy differs (epochs 10/20/30/40, newest state only)')
    output = Path(cfg.output_dir).resolve()
    require(output == (root / 'run' / 'production').resolve(), 'Unexpected output directory')
    require(cfg.resume is None, 'Pass --resume on the command line, not in the YAML')
    if resume is None:
        weights = list((output / 'checkpoints' / 'weights').glob('*.pt'))
        require(not weights, 'Fresh start into an output directory that already holds weights')
        return
    state = Path(resume).resolve()
    require(state.parent == output / 'checkpoints' / 'state' and (state / 'trainer_state.json').is_file(),
            f'Invalid resume state {state}')
    cfg.resume = str(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = OmegaConf.load(config)
    check_config(cfg, config.parent.parent, args.resume)
    runtime.Wan22Trainer = DeltaFlowTrainer
    runtime.run_training(cfg)
