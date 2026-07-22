import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import math
from scipy.optimize import curve_fit, brentq
from helper_functions.state_measurement import *
from scipy.special import erfc
import plotly.graph_objects as go
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
from scipy.optimize import brentq

