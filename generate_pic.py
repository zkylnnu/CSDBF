import matplotlib.pyplot as plt
import numpy as np

def generate_png(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)
    numlabel = np.where(numlabel > 1, 0, 1)
    plt.imshow(numlabel, cmap='gray')
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass