# -*- coding: utf-8 -*-
"""
get latency of the SC

Created on Wed Sep 25 11:18:52 2024

@author: audiobunka
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
from UserModules.pyDPOAEmodule import getSClat



fsamp = 44100
buffersize = 2048


latSC = getSClat(fsamp,buffersize,10)


