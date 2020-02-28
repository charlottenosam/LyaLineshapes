import os

from .utils import *
from .lya_cross_section import *
from .ionizing_sources import *
from .bubble_properties import *
from .opticaldepth import *

# Plotting
from matplotlib import rc_file
rc_file(os.environ['WORK_DIR']+'/code/matplotlibrc')

import matplotlib.pyplot as plt

import matplotlib.mathtext as mathtext
mathtext.FontConstantsBase.sup1 = 0.5
mathtext.FontConstantsBase.sub1 = 0.2
mathtext.FontConstantsBase.sub2 = 0.2

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (4,4)