import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
h=0.3
tt, yy = np.meshgrid(np.arange(-5, 5, h), np.arange(-5, 5, h))
t = np.c_[tt.ravel(), yy.ravel()]

ff2=plt.figure()
def animate(i):
	f=lambda h, j: np.sin(h*i)+np.cos(j)*i
	z=np.zeros((tt.shape[0] ** 2, 1))
	k=0
	for u, j in t:
		z[k, 0]=f(u, j)
		k+=1
	z=z.reshape(tt.shape)

	cont = plt.contourf(tt, yy, z, 25,cmap=plt.cm.rainbow)
	print(i)
	return cont

anim = anim.FuncAnimation(ff2, animate, frames=60)
anim.save('animation.mp4', fps=30)
plt.show()

input()