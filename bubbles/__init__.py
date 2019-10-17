import os

from .utils import *
from .lya_cross_section import *
from .ionizing_sources import *

# Plotting
from matplotlib import rc_file
rc_file(os.environ['WORK_DIR']+'/code/matplotlibrc')

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (4,4)