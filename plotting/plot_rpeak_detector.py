import numpy as np
import matplotlib.pyplot as plt


def plot_artefacts_on_signal(times, values, artefacts):
    fig, ax = plt.subplots()

    # plot the signal
    ax.plot(times, values, label='Signal')

    # plot vertical lines for the artefacts
    for artefact in artefacts:
        ax.vlines(artefact, min(values), max(values), colors='r', linestyles='dotted', label='Artefact')

    # remove duplicates in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Display the plot
    plt.show()
