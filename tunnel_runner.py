import typing as tp
from pathlib import Path


DATA_DIR: Path = Path('.').resolve() / 'data'
PLOTS_DIR: Path = DATA_DIR / 'plots'
MODEL_DIR: Path = DATA_DIR / 'model'


def create_required_directories():
    DATA_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


if __name__ == '__main__':
    print('Hello World!')

    create_required_directories()
    print('Made directories!')
