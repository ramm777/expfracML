

#-----------------------------------------------------------------------------------------------------------------------
# How to save plot in linux without showing it
import matplotlib
matplotlib.use('Agg') # no UI backend

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)
plt.title('About as simple as it gets, folks')

#plt.show()
plt.savefig("matplotlib.png")  #savefig, don't show


#-----------------------------------------------------------------------------------------------------------------------
# Save numpy array as text
np.savetxt(path_results / "results.txt", np.array(losses_all), delimiter=',', fmt="%s")