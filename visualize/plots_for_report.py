import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_bias(btm=-6, top=0.5, left=-100, right=10100, epochs=[1, 5, 15, 25]):

	max_epoch = max(epochs)
	cm_subsection = np.linspace(0, 1, len(epochs))
	colors = [cm.coolwarm(x) for x in cm_subsection]

	nsubplots = len(epochs)

	data = []
	for epoch in epochs:
		data.append(np.loadtxt('bias_' + str(epoch) + '.out.out'))

	fig = plt.figure(figsize=(8, 8))
	fig.suptitle('Token Biases', fontsize=14)

	for i in range(nsubplots):
		d = data[i]
		ax = fig.add_subplot(nsubplots,1,i+1)
		ax.set_ylim([btm, top])
		ax.set_xlim([left, right])
		ax.set_xlabel('Token ID')
		ax.set_ylabel('Bias')

		d = data[i]#data[len(data)-i-1]
		ax.plot([i for i in range(10000)], d, '.', color=colors[i], label='epoch ' + str(epochs[i]))
		ax.legend(loc='lower right')

	return fig



def plot_lr_and_nos(btm=0, top=9, left=0, right=25):

	pad = 2

	nos3 = [7.80085176,  6.96641636, 6.71770664, 6.54772823, 6.53653886, 6.48148973,6.40676134, 6.34532029, 6.29768111, 6.24928931, 6.21683978, 6.19823665,6.19964024, 6.1560456,  6.1299551,  6.13426572, 6.10083784, 6.087691,6.08645214, 6.07741874, 6.07278962, 6.05239725, 6.05124284, 6.04208099, 6.04465724]
	nos10 = [7.27830526, 6.63526339, 6.57596751, 6.42818835, 6.46787793, 6.24819891, 6.22920343, 6.15329442, 6.16334018, 6.07341079, 6.06116261, 6.02238325, 5.99800341, 5.98866981, 5.99475199, 5.96913276, 5.95193775, 5.94549406, 5.9205521,  5.91665143, 5.91162533, 5.90794881, 5.91375392, 5.92078017, 5.91908026]
	nos30 = [6.93360768, 6.51923198, 6.42640099, 6.33051556 ,6.32102371, 6.1615834, 6.136755 ,  6.06643297, 6.0217904,  6.00837267, 5.97588923, 5.92001436, 5.89324785, 5.86478304, 5.85651481, 5.86266465, 5.8169108 , 5.81306811 ,5.80432393, 5.80724329, 5.78740921, 5.78838661, 5.77892116, 5.77663244, 5.78925113]

	lr02 = nos30
	lr1 = [6.34288781, 6.11154157, 6.02068028, 5.95948488, 5.92415494, 5.96950956, 6.02766776, 6.04477944, 6.09665278, 6.02006309, 6.03435604, 6.06938232, 6.13501875, 6.1236864,  6.13207406, 6.21749371, 6.2196558,  6.22875941, 6.26360372 ,6.23286834, 6.29297181, 6.31245041, 6.38159713, 6.35080525, 6.40126005]
	lr5 = [6.52715731 ,6.7445445,  6.58827615, 6.55212824 ,6.53868465, 6.53123245, 6.52819047, 6.52492091, 6.52395396, 6.52328001 ,6.52273523 ,6.52235897, 6.5220511,6.52184124, 6.52168987 ,6.5215865,  6.5215202 , 6.52148819 , 6.52145473, 6.52143785, 6.5214167 , 6.52140864, 6.52142274, 6.52146094, 6.52146847]

	lr02nobias = [7.83, 7.20, 6.77, 6.32, 6.26, 6.19, 6.16, 6.15, 6.08, 6.03, 6.00, 5.97, 5.97, 5.94, 5.92, 5.88, 5.87, 5.86, 5.84, 5.83, 5.83, 5.83, 5.81, 5.82, 5.82] 
	epochs = [i+1 for i in range(len(lr1))]

	fig = plt.figure(figsize=(12,4))
	#fig.suptitle('Effect of # Samples and Learning Rate')

	ax = fig.add_subplot(1,3,1)
	ax.set_xlim([left, right])
	ax.set_ylim([btm, top])
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Crossentropy')
	ax.plot(epochs, nos3, label='NoS=3')
	ax.plot(epochs, nos10, label='NoS=10')
	ax.plot(epochs, nos30, label='NoS=30')
	ax.annotate('# Samples', xy=(0.5, 1), xytext=(0, 3*pad),
		xycoords='axes fraction', textcoords='offset points',
		size='large', ha='center', va='baseline')

	ax.legend(loc='lower right')

	ax = fig.add_subplot(1,3,2)
	ax.set_xlim([left, right])
	ax.set_ylim([btm, top])
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Crossentropy')
	ax.plot(epochs, lr02, label='LR=0.2')
	ax.plot(epochs, lr1, label='LR=1')
	ax.plot(epochs, lr5, label='LR=5')

	ax.legend(loc='lower right')

	ax.annotate('Learning Rate', xy=(0.5, 1), xytext=(0, 3*pad),
		xycoords='axes fraction', textcoords='offset points',
		size='large', ha='center', va='baseline')

	ax = fig.add_subplot(1,3,3)
	ax.set_xlim([left, right])
	ax.set_ylim([btm, top])
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Crossentropy')
	ax.plot(epochs, lr02, label='w/ bias')
	ax.plot(epochs, lr02nobias, label='no bias')

	ax.legend(loc='lower right')

	ax.annotate('Bias', xy=(0.5, 1), xytext=(0, 3*pad),
		xycoords='axes fraction', textcoords='offset points',
		size='large', ha='center', va='baseline')

	return fig

def plot_ours_vs_awd(left=0, right=40, btm=0, top=9):


	awd = [6.62, 6.49, 6.40, 6.34, 6.29, 6.25, 6.23, 6.21, 6.18, 6.17, 6.16, 6.13, 6.11, 6.08, 6.07, 6.06, 6.04, 6.02, 6.00, 6.00, 5.97, 5.96, 5.95, 5.93, 5.94, 5.91, 5.89, 5.88, 5.86, 5.85, 5.84, 5.85, 5.82, 5.81, 5.80, 5.79, 5.78, 5.77, 5.78, 5.79]
	our = [7.94, 7.27, 6.79, 6.66, 6.61, 6.45, 6.41, 6.38, 6.34, 6.32, 6.27, 6.26, 6.21, 6.19, 6.20, 6.14, 6.13, 6.12, 6.11, 6.09, 6.05, 6.04, 6.03, 6.02, 6.02, 6.00, 6.00, 5.99, 5.98, 5.97, 5.96, 5.96, 5.95, 5.95, 5.94, 5.94, 5.93, 5.93, 5.92, 5.92]

	awd_adam_em3 = [5.27, 5.10, 5.06, 5.07, 5.10, 5.12, 5.15, 5.20, 5.25, 5.34, 5.38, 5.40, 5.42, 5.44, 5.45, 5.47, 5.49, 5.50, 5.50, 5.52, 5.53, 5.54, 5.55, 5.57, 5.57, 5.58, 5.60, 5.60, 5.62, 5.62, 5.62, 5.63, 5.65, 5.64, 5.65, 5.66, 5.66, 5.67, 5.67, 5.67]
	awd_adam_em4 = [6.19, 5.88, 5.68, 5.53, 5.43, 5.35, 5.28, 5.23, 5.19, 5.15, 5.12, 5.09, 5.06, 5.03, 5.01, 4.99, 4.97, 4.96, 4.94, 4.93, 4.92, 4.91, 4.90, 4.89, 4.88, 4.87, 4.87, 4.87, 4.86, 4.86, 4.85, 4.85, 4.85, 4.85, 4.84, 4.84, 4.84, 4.84, 4.84, 4.84]
	epochs = [i+1 for i in range(right)]

	fig = plt.figure(figsize=(8,4))
	#fig.suptitle('Effect of # Samples and Learning Rate')

	ax = fig.add_subplot(1,2,1)
	ax.set_xlim([left, right])
	ax.set_ylim([btm, top])
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Crossentropy')
	ax.plot(epochs, our, label='ours')
	ax.plot(epochs, awd, label='awd-rnn')
	ax.legend(loc='lower right')

	ax = fig.add_subplot(1,2,2)
	ax.set_xlim([left, right])
	ax.set_ylim([btm, top])
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Crossentropy')
	ax.plot(epochs, awd_adam_em3, label='awd-rnn 1e-3')
	ax.plot(epochs, awd_adam_em4, label='awd-rnn 1e-4')
	ax.legend(loc='lower right')

	return fig
#fig1 = plot_bias()
#fig2 = plot_lr_and_nos()
fig3 = plot_ours_vs_awd()
#plt.show()
#plt.show()

#fig1.savefig('../../text/report_plots/bias.eps')
#fig2.savefig('lr_and_nos.eps')
fig3.savefig('../../text/report_plots/ours_vs_awd.eps')
