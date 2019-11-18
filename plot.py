"""
This file contains settings for producing plots of the various metrics.
They are imported by the analyze.ipynb notebook, which produces the plots.
"""

# min and max k for x axis
MIN_K = 1
MAX_K = 90000

# Sizes for figures
FIGSIZE_SMALL = (4, 2.0)  # small
FIGSIZE_BIG = (6, 4)  # big

# Specify the colors for the lines
# colors from here: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
MODEL2COLOR = {'fusion': 'orange', 'gpt2': 'green', 'human': 'red'}

# Specify the line styles
# https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/linestyles.html
MODEL2LINESTYLE = {'fusion': 'solid', 'gpt2': 'solid', 'human': 'solid'}

# Specify the thickness of the lines
MODEL2LINEWIDTH = {'fusion': 1, 'gpt2': 1, 'human': 1.5}

# Specify name of model as it should appear in legend
MODEL2LEGENDNAME = {'human': 'Human', 'fusion': 'Fusion Model', 'gpt2': 'GPT2-117'}
