import matplotlib.pyplot as plt
import pandas as pd

name = "../model/fcn/part_BN_outscale_3l_4h_models/plots/noBN_3l_4h"
data = pd.read_csv(name+".txt", header=None, sep=' ', dtype='float')

plt.plot(data.index, data[1], 'y*-')
plt.plot(data.index, data[2], 'r--')
plt.savefig(name+"_plot.png")
plt.show()
