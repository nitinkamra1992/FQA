import numpy as np
import os

disp_avlbl = True
from os import environ
if 'DISPLAY' not in environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotCurve(values, timesteps=None, errs=None, savefile=None, fontsize=None, **kwargs):
    ''' Plots values vs iterations

    Args:
        values: Values to plot (y-axis)
        timesteps: X-axis labels
        errs: Y-axis errors to plot errorbars
        savefile: If not None, plot is saved to this file
        fontsize: Font size for plot labels
    '''
    # Match array lengths
    n_steps = len(values)
    if timesteps is not None:
        assert len(values) == len(timesteps)

    # Set font size
    if fontsize is not None:
        default_fontsize = plt.rcParams.get('font.size')
        plt.rcParams.update({'font.size': fontsize})

    # Plot curves
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    if errs is None:
        ax.plot(values, marker='o')
    else:
        ax.errorbar(range(n_steps), values, yerr=errs, fmt='o-')

    # Labels
    xticks = timesteps if timesteps is not None else range(1, n_steps+1)
    plt.xticks(range(n_steps), xticks)
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else 'Step')
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else 'Value') 

    # Save/display figure
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    # Reset font size
    if fontsize is not None:
        plt.rcParams.update({'font.size': default_fontsize})
    plt.close()


def plotCurves(curves, timesteps=None, errs=None, legend_labels=None, savefile=None, fontsize=None, **kwargs):
    ''' Plots multiple curves sharing their x-axis

    Args:
        curves: List of curves to plot (y-axis)
        timesteps: X-axis labels
        errs: List of Y-axis errs for each curve in curves
        legend_labels: Legends for each curve
        savefile: If not None, plot is saved to this file
        fontsize: Font size for plot labels
    '''
    # Match lengths
    num_curves = len(curves)
    if legend_labels is not None:
        assert len(legend_labels) == num_curves
    if errs is not None:
        assert num_curves == len(errs)

    n_steps = len(curves[0])
    for curve in curves:
        assert n_steps == len(curve)
    if timesteps is not None:
        assert n_steps == len(timesteps)
    if errs is not None:
        for err in errs:
            assert n_steps == len(err)  

    # Set font size
    if fontsize is not None:
        default_fontsize = plt.rcParams.get('font.size')
        plt.rcParams.update({'font.size': fontsize})

    # Plot curves
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    handles = []

    if 'yscale' in kwargs:
        # ax.set_yscale("log", nonposy='clip')
        ax.set_yscale(kwargs['yscale'], nonposy='clip')
    
    for i in range(num_curves):     
        label = legend_labels[i] if legend_labels is not None else None
        if errs is None:
            h, = ax.plot(curves[i], marker='o', label=label)
        else:
            h, _, _ = ax.errorbar(range(n_steps), curves[i], yerr=errs[i], fmt='o-', label=label)
        handles.append(h)

    # Labels
    xticks = timesteps if timesteps is not None else range(1, n_steps+1)
    plt.xticks(range(n_steps), xticks)
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else 'Step')
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else 'Value') 
    if legend_labels is not None:
        # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # plt.legend(handles=handles)
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Save/display figure
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    # Reset font size
    if fontsize is not None:
        plt.rcParams.update({'font.size': default_fontsize})
    plt.close()


def curve_plot(values_dict):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for key, values in values_dict.items():
        if values == []:
            continue
        ax.plot(values, label=key)
        ax.set_xlabel('epochs')
        ax.set_title('plot')
        ax.legend()
    return fig


def create_curve_plots(name, plot_dict, log_dir):
    fig = curve_plot(plot_dict)
    fig.suptitle(name)
    fig.savefig(os.path.join(log_dir, name + '_curve.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)