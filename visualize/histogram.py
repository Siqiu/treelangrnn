import glob
import numpy as np 
import itertools
import matplotlib.pyplot as plt
import matplotlib

#path = '../results/lstm_results/adam-0.001-0.4-distance-10.out'
#data = np.loadtxt(path)
#plt.hist(data)

nbins = 100
basepath = '../results/lstm_results/'

# generate all paths
'''
files = [f for f in glob.glob(basepath + "*.out")]

# range of values
d, D, e, E = 1e5, 1e-5, 1e5, 1e-5
for f in files:

	data = np.loadtxt(f)

	if 'distance' in f:
		d = min(d, np.amin(data))
		D = max(D, np.amax(data))
	elif 'entropy' in f:
		e = min(e, np.amin(data))
		E = max(E, np.amin(data))
	else:
		pass


# bins (hardcoded!)
d, D = -25, 25
e, E = 0, 20
dbins = np.linspace(d, D, nbins)
ebins = np.linspace(e, E, nbins)

settings = [['adam', 0.001], ['adam', 0.0005], ['adam', 0.0001]]
			#['sgd', 1.0], ['sgd', 10.0], ['sgd', 30.0]]
epochs = [5, 10, 15, 20]
files = list(itertools.product(*[settings, epochs]))
print(files)


# plot histogram
for i, (opt, lr) in enumerate(settings):


	for j, epoch in enumerate(epochs):

		d_path = basepath + opt + '-' + str(lr) + '-0.4-distance-' + str(epoch) + '.out'
		e_path = basepath + opt + '-' + str(lr) + '-0.4-entropy-' + str(epoch) + '.out'

		distance = np.loadtxt(d_path)
		entropy = np.loadtxt(e_path)

		plt.subplot(4, 2, 2*j + 1)
		plt.ylim([0, 12000])
		plt.hist(distance, dbins)
		plt.axvline(distance.mean(), color='k', linestyle='dashed', linewidth=2)

		plt.subplot(4, 2, 2*j + 2)
		plt.ylim([0, 12000])
		plt.hist(entropy, ebins)
		plt.axvline(entropy.mean(), color='k', linestyle='dashed', linewidth=2)
		print([distance.mean(), entropy.mean()])
	
	plt.show()

'''


from scipy.interpolate import interpn
def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
	"""
	Scatter plot colored by 2d histogram
	"""
	if ax is None :
		fig , ax = plt.subplots()
	data , x_e, y_e = np.histogram2d( x, y, bins = bins)
	z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

	# Sort the points by density, so that the densest points are plotted last

	if sort :
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

	ax.scatter( x, y, c=z, cmap='magma', **kwargs)
	return ax

class EntropyDistance2D:

	def __init__(self, base, optimizer, learning_rate, regularizer, epochs):

		self.epochs = epochs

		self.distance_bases = [base + 'adam-0.0001-0.0-distance-', base + 'sgd-10.0-0.0-distance-']
		self.entropy_bases = [base + 'adam-0.0001-0.0-entropy-', base +  'sgd-10.0-0.0-entropy-']


	def make_heatmap(self, dleft, dright, eleft, eright, nbins):

		nrows = len(self.epochs)
		ncols = 2

		self.fig = plt.figure(figsize=(8, 8))
		self.fig.suptitle('Similarity-Entropy Scatter', fontsize=14)


		for i, (distance_base, entropy_base) in enumerate(zip(self.distance_bases, self.entropy_bases)): 
			
			for j, epoch in enumerate(self.epochs):

				distance_path = distance_base + str(epoch) + '.out'
				entropy_path = entropy_base + str(epoch) + '.out'

				distance = np.loadtxt(distance_path)
				entropy = np.loadtxt(entropy_path)

				idx = 2*(j+1) - i
				ax = self.fig.add_subplot(nrows, ncols, idx)
			
				ax.set_ylim([eleft, eright])
				ax.set_xlim([dleft, dright])
				ax.set_aspect('equal')
				ax.set_ylabel('Entropy')
				ax.set_xlabel('Similarity')

				if j == 0: ax = self._make_col_title(ax, 'adam' if i == 0 else 'sgd')
				density_scatter(distance, entropy, ax=ax, bins=nbins)

	def _make_col_title(self, ax, title, pad=5):
		ax.annotate(title, xy=(0.5, 1), xytext=(0, 3*pad),
				xycoords='axes fraction', textcoords='offset points',
				size='large', ha='center', va='baseline')
		return ax

	def save(self, path):
		self.fig.savefig(path)

class EntropyHistogram:

	def __init__(self, base, epochs):

		self.epochs = epochs

		self.entropie_paths = [base + 'entropy_' + str(epoch) + '.out' for epoch in epochs]
		self.entropy = dict()
		for epoch, path in zip(epochs, self.entropie_paths):
			self.entropy[epoch] = np.loadtxt(path)

		self.valid_path = base + 'val_loss.out.out'
		#self.valid = np.loadtxt(self.valid_path)
		#print(self.valid.shape)

		self.title = 'Histogram of Entropy by Epoch'


	def make_histogram(self, eleft, eright, nbins, ylow=0, ytop=25000):

		nrows = 1
		ncols = len(self.epochs)
		ebins = np.linspace(eleft, eright, nbins)

		self.fig = plt.figure(figsize=(14, 4))
		#self.fig = plt.figure(figsize=(2, 8))
		#self.fig.suptitle(self.title, fontsize=14)

		for i, epoch in enumerate(self.epochs):

			# plot distance histo
			'''
			ax = self.fig.add_subplot(nrows, ncols, 2*i + 1)
			ax.set_ylim([ylow, ytop])

			ax = self._make_row_title(ax, epoch)
			if i == 0: ax = self._make_col_title(ax, 'Similarity')

			ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax.hist(self.distance[epoch], dbins, edgecolor='black', color='white')
			ax.axvline(self.distance[epoch].mean(), color='r', linestyle='dashed', label='avrg. similarity')
			ax.legend(loc='upper left')
			ax.set_xlabel('exp(-d(c,c+)+b)')
			ax.set_ylabel('count')
			'''

			# plot entropy histo

			ax = self.fig.add_subplot(nrows, ncols, i+1)
			ax.set_ylim([ylow, ytop])

			#ax = self._make_col_title(ax, epoch)

			ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax.hist(self.entropy[epoch], ebins, edgecolor='black', color='white')
			ax.axvline(np.mean(self.entropy[epoch]), color='r', linestyle='dashed', label='avrg. entropy')
			ax.legend(loc='upper right')
			ax.set_xlabel('Entropy (Epoch: ' + str(epoch) + ')')
			if i == 0: ax.set_ylabel('Counts')

	def save(self, path):
		self.fig.savefig(path)

	def _make_row_title(self, ax, epoch, pad=5):
		row_title = 'Epoch:   ' + str(epoch) if epoch < 10 else 'Epoch: ' + str(epoch)
		ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')
		return ax

	def _make_col_title(self, ax, title, pad=5):
		ax.annotate(title, xy=(0.5, 1), xytext=(0, 3*pad),
				xycoords='axes fraction', textcoords='offset points',
				size='large', ha='center', va='baseline')
		return ax


eh = EntropyHistogram('./', [20,20,40,40])
eh.make_histogram(0, 40, 50)
plt.show()
#eh.save('../../text/report_plots/awd_adam_0001.eps')
#ed2d = EntropyDistance2D('../results/lstm_results/', 'adam', .0001, 0.4, [5, 10, 15, 20])
#ed2d.make_heatmap(-25, 25, 0, 30, 100)
#ed2d.save('similarity-entropy-scatter.png')