import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

R = 3
T = 10
time = 3 * T
slow_motion_factor = 1
fps = 60
interval = 1 / fps

fig = plt.figure(figsize=(7.2, 7.2))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.set_xlim(-1.5 * R, 1.5 * R)
ax.set_ylim(-1.5 * R, 1.5 * R)

runner = plt.Circle((0, 0), 0.1, fc='r')
circle = plt.Circle((0, 0), R, color='b', fill=False)
ax.add_artist(circle)
time_text = ax.text(1.1 * R, 1.1 * R,'', fontsize=15)

def init():
    time_text.set_text('')
    return time_text,


def animate(i):
    if i == 0:
        ax.add_patch(runner)
    t = i * interval
    time_text.set_text('{:1.2f}'.format(t))
    x = R * np.sin(2 * np.pi * t / T)
    y = R * np.cos(2 * np.pi * t / T)
    runner.center = (x, y)
    return runner, time_text

anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=time * fps,
    interval=1000 * interval * slow_motion_factor,
    blit=True,
)

plt.show()