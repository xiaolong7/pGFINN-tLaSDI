import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from copy import deepcopy
plt.rcParams["text.usetex"] = False

def max_err_heatmap(max_err, p1_test, p2_test, data_path, idx_list=[], idx_param=[],
                    xlabel='param1', ylabel='param2', label='Max. Relative Error (%)', 
                    dtype='int', scale=1, vmin=None, vmax=None):
    sns.set(font_scale=1.3)
    if dtype == 'int':
        max_err = max_err.astype(int)
        fmt1 = 'd'
    else:
        fmt1 = '.1f'
    rect = []
    for i in range(len(idx_param)):
        print(f"idx: {idx_param[i][0]}, param: {idx_param[i][1]}")
        idd = idx_param[i][0]
        rect.append(
            patches.Rectangle((idx_list[idd, 1], idx_list[idd, 0]), 1, 1, linewidth=2, edgecolor='k', facecolor='none'))
    rect2 = deepcopy(rect)

    if max_err.size < 100:
        fig = plt.figure(figsize=(5, 5))
    else:
        fig = plt.figure(figsize=(9, 9))

    fontsize = 14

    # Create annotation strings based on value thresholds
    annot_data = np.empty_like(max_err, dtype=object)
    for i in range(max_err.shape[0]):
        for j in range(max_err.shape[1]):
            val = max_err[i, j] * scale
            if val >= 10:
                annot_data[i, j] = f"{round(val)}"
            else:
                annot_data[i, j] = f"{val:.1f}"
                
    ax = fig.add_subplot(111)
    cbar_ax = fig.add_axes([0.99, 0.19, 0.02, 0.7])

    # vmax = max_err.max() * scale
    if vmin == None: vmin = max_err.min()
    if vmax == None: vmax = max_err.max()
    heatmap = sns.heatmap(max_err * scale, ax=ax, square=True, 
                        xticklabels=p2_test, yticklabels=p1_test, 
                        annot=annot_data, annot_kws={'size': fontsize}, fmt='', 
                        cbar_ax=cbar_ax, cbar=True, cmap='vlag', 
                        robust=True, vmin=vmin, vmax=vmax)

    # Define a formatter function to add the percentage sign
    def percentage_formatter(x, pos):
        if dtype == 'int':
            return "{:.0f}%".format(x)
        else:
            return "{:.1f}%".format(x)
            
    # Apply the formatter to the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    for i in rect2:
        ax.add_patch(i)

    # format text labels
    fmt = '{:0.2f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Width ($\\omega$)', fontsize=24)
    ax.set_ylabel('Amplitude ($\\alpha$)', fontsize=24)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    plt.tight_layout()
    if label == 'Residual Norm':
        plt.savefig(data_path + 'heatmap_resNorm.png', bbox_inches='tight')
    else:
        plt.savefig(data_path + 'heatmap_maxRelErr_tlasdi.png', bbox_inches='tight')
    plt.show()