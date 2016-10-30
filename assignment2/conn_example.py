from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from scipy.misc import imread

book = imread('book.jpg')
fig = plt.figure(1, figsize=(6,3))
ax1 = plt.subplot(121)
ax1.imshow(book)

ax2 = plt.subplot(122)
#xyA=(0.7, 0.7)
xy=(1, 1)
coordsA="data"
coordsB="data"
con = ConnectionPatch(xyA=(0.1,0.1), xyB=(100,100), coordsA=coordsA, coordsB=coordsB,
                      axesA=ax2, axesB=ax1,
                      arrowstyle="->", shrinkB=5)
ax2.add_artist(con)


plt.draw()
plt.show()
