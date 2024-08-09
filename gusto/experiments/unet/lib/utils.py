from pathlib import Path

LOGDIR = Path(__file__).parent.parent.joinpath('logs')
CHECKPOINT_DIR = LOGDIR.parent.joinpath('checkpoints')
