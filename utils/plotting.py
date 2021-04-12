import matplotlib.pyplot as plt
import cv2


def plot_to_axe(ax, image, title):
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray', interpolation='nearest')
    else:
        ax.imshow(image)
    ax.set_title(title, color='red', fontsize=50)


def plot_both(image1, image2, title1, title2,
              show_plot=True, save_plot=False, output_fname=None):
    if (not show_plot) and (not save_plot):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    fig.tight_layout()

    plot_to_axe(ax1, image1, title1)
    plot_to_axe(ax2, image2, title2)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if show_plot:
        plt.show()
    if save_plot:
        if output_fname:
            fig.savefig(output_fname, dpi=fig.dpi)
    return fig


def plot_four(image11, image12, image21, image22, title11, title12, title21, title22,
              show_plot=True, save_plot=False, output_fname=None):
    if (not show_plot) and (not save_plot):
        return
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(24, 9))
    fig.tight_layout()

    plot_to_axe(ax11, image11, title11)
    plot_to_axe(ax12, image12, title12)
    plot_to_axe(ax21, image21, title21)
    plot_to_axe(ax22, image22, title22)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if show_plot:
        plt.show()
    if save_plot:
        if output_fname:
            fig.savefig(output_fname, dpi=fig.dpi)
    return fig
