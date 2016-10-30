import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax1.scatter([5, 7], [10, 2])
ax2 = fig.add_subplot(1,2,2)
ax2.scatter([5, 7], [10, 2])

coord1 = ax1.transData.transform([7,2])
coord2 = ax2.transData.transform([5,10])
line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]))
fig.lines = line,
coord1 = ax2.transData.transform([7,2])
coord2 = ax1.transData.transform([5,10])
line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]))
fig.lines = line,

plt.show()