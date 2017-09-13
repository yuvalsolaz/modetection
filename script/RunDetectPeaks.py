



import numpy as np
from detect_peaks import detect_peaks
x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5


# set minimum peak height = 0 and minimum peak distance = 20
detect_peaks(x, mph=0, mpd=20, show=True)