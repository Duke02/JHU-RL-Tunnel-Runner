from pathlib import Path
import typing as tp
import argparse

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from environment import Environment


def get_arguments() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--q-table', help='Path to Q Table. Must be saved through NumPy savetxt function', type=str, required=True)

    args: argparse.Namespace = parser.parse_args()

    args.q_table = Path(args.q_table).resolve()

    if not args.q_table.exists():
        raise argparse.ArgumentError('--q-table', 'Path to Q-Table could not be resolved!')

    return args


def plot_locations(e: Environment, q: np.ndarray) -> tp.Iterable[np.ndarray]:
    done: bool = False

    state: int = e.reset(seed=13)

    while not done:
        action: int = np.argmax(q[state, :])
        next_state, reward, terminated = e.execute_action(action)
        done = terminated
        state = next_state
        yield e.get_map()


plt.rcParams['patch.force_edgecolor'] = True
fig: Figure = plt.figure()  # , ax = plt.subplots(1, 1)
ax = fig.add_subplot(111)

env: Environment = Environment()

arguments: argparse.Namespace = get_arguments()
q_table: np.ndarray = np.loadtxt(arguments.q_table)

plot_maps = [m for m in plot_locations(env, q_table)]


def animate(plot_index: int):
    print(f'Animating at Index {plot_index}')
    ax.cla()
    cmap: LinearSegmentedColormap = LinearSegmentedColormap('custom_cmap', {
        'red': [(0, 1, 1),
                (.1, 0, 0),
                (.45, 0, 0),
                (.65, 0, 0),
                (1, 1, 1)],
        'green': [(0, 1, 1),
                  (.1, 0, 0),
                  (.45, 1, 1),
                  (.65, 0, 0),
                  (1, 0, 0)],
        'blue': [(0, 1, 1),
                 (.1, 0, 0),
                 (.45, 1, 1),
                 (.65, 0, 0),
                 (1, 0, 0)]
    })

    ax.pcolor(np.rot90(plot_maps[plot_index].T), cmap=cmap, edgecolor='b')
    # ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
    return fig


print(f'Rendering video...')

animator: FuncAnimation = FuncAnimation(fig, animate, cache_frame_data=True, save_count=len(plot_maps))
animator.save(Path('.').resolve() / 'data' / 'plots' / 'animation.mp4', codec='mpeg4', writer='ffmpeg', fps=1, dpi=125)
