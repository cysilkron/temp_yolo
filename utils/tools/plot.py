import matplotlib.pyplot as plt 
from .general import *

def plt_bar_dist(value_counts, title=None):
    fig, ax = plt.subplots()

    if isinstance(value_counts, pd.Series):
        ax.bar(value_counts.index, value_counts.values)
    elif isinstance(value_counts, dict):
        ax.bar(value_counts.keys(), value_counts.values())

    if title:
        ax.set_title(title)
