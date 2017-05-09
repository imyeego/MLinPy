# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(h):
	return 1.0 / (1.0 + np.exp(-h))

h = np.arange(-10, 10, 0.1)
s_h = sigmoid(h)
plt.plot(h, s_h)
plt.axvline(0.0, color = 'k')
plt.axhspan(0.0, 1.0, facecolor = '1.0', alpha = 1.0, ls = 'dotted')
plt.axhline(0.5, ls = 'dotted', color = 'k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('h')
plt.ylabel('$S(h)$')
plt.show()