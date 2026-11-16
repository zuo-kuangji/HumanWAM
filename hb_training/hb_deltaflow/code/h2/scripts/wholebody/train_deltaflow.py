"""ImageWAM Delta-flow training entry (G1 and H2, augmentation v2), installed into every Delta-flow code tree.
Derived from train_a3a7.py:
- same native runtime/trainer, same 32-GPU loader-length checks and finish receipts;
- no separate 1-update smoke phase (the supervisor retries / resumes instead);
- --resume <run/production/checkpoints/state/step_N> continues an interrupted run from its newest state.
Schedules accepted (checked against the run's data-verified.json):
- 40 epochs, saves at epochs 10/20/30/40, keep 4 weight files (the original 2026-11-09 protocol);
- 2026-11-15: fixed optimizer steps (data-verified.json fixed_steps=true): max_steps / warmup / save_steps exactly as
  recorded at preparation (8000 steps, 6000/8000 or 4000/8000; C6 16000), newest state only.
Training episodes: exactly 0..N-1 of the collection as recorded at preparation (2026-11-14 user: all episodes).
Datasets: the HB aug v2 loaders, and (2026-11-15, H2 training guides) the original A18 loader H2PourDataset with its
own visual augmentation b6-visual-aug-v2 for runs prepared with recipe 'legacy-a18-e40'.
"""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from imagewam import runtime
from imagewam.trainer import Wan22Trainer

# dataset class -> required train_split_mode (None: the loader itself admits exactly its episode set)
DATASET_TARGETS = {
    'imagewam.datasets.wholebody_hbaug.WholeBodyHBAugV2LeRobotDataset': 'first_100',   # G1
    'imagewam.datasets.h2_hbaug.H2HBAugV2Dataset': None,                               # H2, aug v2
    'imagewam.datasets.h2_pour.H2PourDataset': None,                                   # H2, legacy a18 e40 recipe
}
AUG_VERSION = 'hb-aug-v2-20261109'
LEGACY_TARGET, LEGACY_AUG = 'imagewam.datasets.h2_pour.H2PourDataset', 'b6-visual-aug-v2'
EPOCHS, SAVE_EVERY_EPOCHS, WARMUP_EPOCHS, WORLD = 40, 10, 2, 32
EXPECTED = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class DeltaFlowTrainer(Wan22Trainer):
    def _resume_or_load_checkpoint(self):
        require(len(self.train_dataset) == EXPECTED['frames'] and len(self.train_loader) == EXPECTED['updates_per_epoch']
                and self.accelerator.num_processes == WORLD,
                'Expected the pinned collection, its updates/epoch and 32 GPUs')
        require(type(self.train_dataset).__name__ in {t.rsplit('.', 1)[1] for t in DATASET_TARGETS},
                'Unexpected dataset class')
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
    EXPECTED.update(frames=int(receipt['frames']), updates_per_epoch=u)
    d = cfg.data.train
    legacy = receipt.get('recipe') == 'legacy-a18-e40'
    require(d._target_ in DATASET_TARGETS and d.image_augmentation.enabled is True
            and (d.image_augmentation.version == AUG_VERSION
                 or (legacy and d._target_ == LEGACY_TARGET and d.image_augmentation.version == LEGACY_AUG)),
            'Config does not use augmentation v2 (or the recorded legacy a18 recipe)')
    split_mode = DATASET_TARGETS[d._target_]
    train_ids = list(receipt['train_episode_ids'])
    require((split_mode is None or d.train_split_mode == split_mode) and train_ids == list(range(len(train_ids)))
            and len(train_ids) > 0, 'Training must use exactly episodes 0..N-1 of the collection')
    if receipt.get('fixed_steps'):
        total = int(receipt['max_steps'])
        require(cfg.max_steps == total and cfg.lr_scheduler_total_steps == total
                and cfg.warmup_steps == int(receipt['warmup_steps']),
                'Fixed-step schedule differs from data-verified.json')
        require(list(cfg.save_steps) == list(receipt['save_steps']) and cfg.save_every == 0 and cfg.save_at_end is True
                and cfg.keep_latest_state_only is True and int(cfg.keep_latest_weights) >= len(receipt['save_steps']),
                'Checkpoint policy differs from data-verified.json (fixed steps, newest state only)')
    else:
        total = EPOCHS * u
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
        init = receipt.get('init_weights')  # HB 2026-11-15: weights-only continuation (fresh optimizer / LR / step)
        if init:
            p = Path(init['path']).resolve()
            require(p.is_file() and p.suffix == '.pt' and p.stat().st_size == int(init['bytes']),
                    f'init weights {p} missing or not the recorded file')
            cfg.resume = str(p)  # a weights file: Wan22Trainer loads it before accelerator.prepare
            print('HB_INIT_WEIGHTS', p, flush=True)
        return
    state = Path(resume).resolve()
    require(state.parent == output / 'checkpoints' / 'state' and (state / 'trainer_state.json').is_file(),
            f'Invalid resume state {state}')
    cfg.resume = str(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--check-only', action='store_true', help='validate the config against data-verified.json, no training')
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = OmegaConf.load(config)
    check_config(cfg, config.parent.parent, args.resume)
    if args.check_only:
        print('CONFIG_OK', config, flush=True)
        raise SystemExit(0)
    runtime.Wan22Trainer = DeltaFlowTrainer
    try:  # 2026-11-15 user: in-training open-loop action MSE every 250 steps (HB_INLOOP_EVERY=0 = off)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'hb_deltaflow'))
        import hb_inloop_deltaflow
        hb_inloop_deltaflow.install(DeltaFlowTrainer, cfg, config.parent.parent)
    except Exception as e:  # noqa: BLE001 - never block training
        print(f'HB_INLOOP disabled: {e!r}', flush=True)
    runtime.run_training(cfg)
