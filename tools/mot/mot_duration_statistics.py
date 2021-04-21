# compute statistics information from MOT results.
import motmetrics as mm
import pandas as pd
import matplotlib.pyplot as plt
from motmetrics.io import Format
import seaborn

class StatisticsProducer():
    def __init__(self, name_file_tuple):
        self.name_file_tuple = name_file_tuple

    def load_and_compute_all(self):
        self.fig, axes = plt.subplots(1, len(self.name_file_tuple), sharey=True)
        plt.subplots_adjust(wspace=.01)
        for (name, file_path), ax in zip(self.name_file_tuple, axes):
            ax.set_title(name)
            ax.set_yscale('log')
            ax.set_xlim([0, 450])
            ax.set_ylim(top=300)
            load_and_compute_statistics(name, file_path, ax)

    def save(self, file_path, **args):
        self.fig.savefig(file_path, **args)

    def show(self):
        plt.show()

def load_and_compute_statistics(name, file_path, ax):
    data = mm.io.loadtxt(file_path, fmt=Format.MOT16)
    grouped = data.groupby(by='Id')
    duration_ids = pd.DataFrame(data={'duration': [], 'id': []})
    for name, group in grouped:
        duration = group.count().max()
        duration_ids = duration_ids.append({'duration':duration, 'id':name}, ignore_index=True)
    # , element="step", fill=False
    # compute average
    mean = duration_ids['duration'].mean()
    std = duration_ids['duration'].std()
    ax.axvline(mean, color='red', linestyle='--', alpha=0.7)
    ax.axvspan(mean-std, mean+std, facecolor='green', alpha=0.2)
    seaborn.histplot(data = duration_ids, x='duration', binwidth=20, ax=ax)
