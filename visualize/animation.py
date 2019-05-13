import glob, os
import numpy as np
from dump import load_contexts

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


class Animation:

	def __init__(self, dir, snapshot_mode = False, xlim=None, ylim=None):
		
		# determine all files
		self.files = [file for file in glob.glob(dir + "context_dump_*")]

		# extract all epoch numbers
		self.epochs = [os.path.splitext(file)[0] for file in self.files]
		self.epochs = [int(s.split('_')[-1]) for s in self.epochs]

		# sort both lists by epoch
		self.epochs, self.files = (list(t) for t in zip(*sorted(zip(self.epochs, self.files))))

		# load data for each epoch
		self.data = np.array([load_contexts(file) for file in self.files])

		# limits
		if xlim is None:
			self.xlim = [-0.5, 0.5]
		else:
			self.xlim = xlim

		if ylim is None:
			self.ylim = [-0.5, 0.5]
		else:
			self.ylim = ylim

		self.fig, self.ax = plt.subplots()

		
		if not snapshot_mode:
			self.ani = FuncAnimation(self.fig, self._draw_next_snapshot, frames=len(self.files), interval=10,
										init_func=self._setup_plot, blit=False, repeat=True)
		else:
			self._setup_plot()

		self.colors = ['k', 'r', 'b', 'g']


	def animate(self):
		pass

	def show(self):
		plt.show()

	def save(self, path):
		pass

	def _setup_plot(self):
		self._clear(0)
		self.plot =	self.ax.plot()
		return self.plot,

	def _draw_next_snapshot(self, i):

		self._clear(self.epochs[i])

		data = self.data[i,:,:]
		max_depth = np.amax(data[:,1])
		hsz = (np.size(data, 1) - 2) // 2

		line_color_index = 0
		for i in range(np.size(data,0) - 16*10, np.size(data, 0)):

			if data[0,0] == 0:
				line_color = self.colors[line_color_index]
				#line_color_index = (line_color_index + 1) % len(self.colors)

			c2 = plt.cm.Greys(float((max_depth-data[i,1]))/max_depth)
			if data[i,0] == 0:
				c1 = plt.cm.Reds(0.75)
			else:
				c1 = plt.cm.Greys(float((max_depth-data[i,0]))/max_depth)

			print(data[i:i+17,:])
			plt.plot([-data[i,2], -data[i,4]], [-data[i,3], -data[i,5]], line_color)
			plt.plot(-data[i,2], -data[i,3], marker='.', markeredgecolor='k', markerfacecolor=c1)
			plt.plot(-data[i,4], -data[i,5], marker='.', markerfacecolor=c2, markeredgecolor='k')


		return self.plot

	def _clear(self, epoch):

		self.ax.cla()
		self.ax.axis(self.xlim + self.ylim)
		self.ax.set_title("Epoch: " + str(epoch))



path = '../results/test/'
a = Animation(path, snapshot_mode=False, xlim=[-1.5,1.5], ylim=[-1.5,1.5])
#a._draw_next_snapshot(60)
#plt.savefig(path + 'snapshot.jpeg')
#a.show()
Writer = writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
a.ani.save('ns_test2.mp4', writer=writer)