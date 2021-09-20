import seaborn as sns






with sns.axes_style("white"):
    ax = sns.heatmap(similarity_array, mask=array_mask, robust=False, annot=True,
                     fmt='.2f', xticklabels=labels, yticklabels=labels)
pic = ax.get_figure()
pic.savefig('heatmap.png')
