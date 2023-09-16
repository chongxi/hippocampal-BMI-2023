from IPython.display import display

import pandas as pd
pd.options.display.max_colwidth = 120
import numpy as np
np.set_printoptions(precision=6, suppress=True)

import seaborn as sns
sns.set_context('paper', font_scale=1)

import warnings
warnings.filterwarnings('ignore', '', )
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

# import matplotlib plt
import matplotlib.pyplot as plt
rc = plt.rcParams
rc['figure.figsize'] = (5, 5)
rc["font.family"] = "Arial"
rc['svg.fonttype'] = 'none'
rc['font.size'] = 6


from matplotlib.colors import LinearSegmentedColormap
neo = LinearSegmentedColormap.from_list('neo', [(0, 0, 0), (0, 1, 0)] , N=100) # black to green

from ipywidgets import interact


def plot_goal(ax, goal_center, goal_radius=15, color='k', fill=False, alpha=1, linewidth=3):
    goal_region = plt.Circle(goal_center, goal_radius, color=color, fill=fill, alpha=alpha, linewidth=linewidth)
    ax.add_patch(goal_region)
    ax.set_xlim([-50,50])
    ax.set_ylim([-50,50])

def plot_trial(t, pos, goal, ax=None, markersize=20, goal_radius=15):
    '''
    plot jumper trials, color coded by time using jet colormap
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(pos[:, 0], -pos[:, 1], c=t, s=markersize, cmap=plt.cm.jet)
    goal_center = (goal[0], -goal[1])
    plot_goal(ax, goal_center, goal_radius)
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    return ax

# plot cumulative distribution function (cdf) of ball velocity
def plot_cdf(data, color, ax=None, fontsize=12):
    data = np.sort(data)
    yvals = np.arange(len(data))/float(len(data))
    if ax is None:
        plt.plot(data, yvals, c=color)
        plt.xlabel('ball velocity (cm/s)', fontsize=fontsize)
        plt.ylabel('CDF', fontsize=fontsize)
    else:
        ax.plot(data, yvals, c=color)
        ax.set_xlabel('ball velocity (cm/s)', fontsize=fontsize)
        ax.set_ylabel('CDF', fontsize=fontsize)
