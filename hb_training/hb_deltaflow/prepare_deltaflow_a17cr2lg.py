"""2026-11-15 23:xx user: Delta Flow aligned with the newest data GR00T trains on. A17 on the CROPPED v2 data (HF
a17-unscrew-bottle-expert-cropped-v2 @3520b70, 100 episodes, 75,184 frames; v1 @407e6e20 with the waits trimmed, same
episodes / features / cameras / task string) with exactly the a17lg recipe (legacy a18 e40 configuration of the H2
training guide: H2PourDataset, b6-visual-aug-v2, no proprio augmentation, 256 x 8000 steps, checkpoints 4000 / 8000,
warmup 400). The task string is unchanged, so the Qwen3 text cache of a17c2 is reused as for a17lg."""
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow_h2legacy as lg  # noqa: E402

lg.TASKS['a17cr2lg'] = dict(
    data=lg.base.P / 'a17-unscrew-bottle-cropped-v2-3520b70' / 'data', frames=75184, code=lg.h2.CODE, action_dim=75,
    qwen_from=lg.base.ROOT / 'a17c2' / 'qwen-text', policy_id='a17-h2-rgb-v1',
    source='humanoid-benchmark/a17-unscrew-bottle-expert-cropped-v2 @ 3520b70f0cc30fec10c5d806167c791cbbfea718')

if __name__ == '__main__':
    lg.prepare('a17cr2lg')
    print('A17CR2LG_DF_OK', flush=True)
