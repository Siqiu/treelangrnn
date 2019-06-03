import numpy as np 
import matplotlib.pyplot as plt
import random
import glob
import torch

#import sys
#sys.path.append("..")

YLIM_HIST = [0, 1.3]
#XLIM_WORDS = [-1.5, 1.5]
#YLIM_WORDS = [-1.5, 1.5]
XLIM_WORDS = [-0.5, 1.]
YLIM_WORDS = [-0.5, 1]
XLIM_EMB = [-1.5, 1.5]
YLIM_EMB = [-1.5, 1.5]


def plot_line(x, y, clr, ls=None, ax=None):
	if ax is None:
		plt.plot(x, y, clr, LineWidth=0.2, ls=ls)
	else:
		ax.plot(x, y, clr, LineWidth=0.2, ls=ls)

def plot_all_word_embeddings(coords, words, xlim, ylim, clr, ax=None):

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	#ax.set_yticks([-1, 0, 1])
	#ax.set_xticks([-1, 0, 1])
	ax.set_xticks([0, 0.5])
	ax.set_yticks([0, 0.5])

	for i in range(len(words)):
		ax.plot(coords[i][0], coords[i][1], clr + 'x')
		ax.annotate(words[i], xy=(coords[i,0]+0.01, coords[i,1]+0.01), fontsize=10)
	return ax

def plot_sequence(data, start, end, xlim, ylim, clr, ls=None, ax=None):

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	ax.set_yticks([-1, 0, 1])
	ax.set_xticks([-1, 0, 1])

	ax.scatter(0, 0, facecolors='none', edgecolors='k')
	plot_line([0, data[start, 1]], [0, data[start, 2]], clr, ax=ax)

	for i in range(start, end-1):
		plot_line([data[i][1], data[i+1][1]], [data[i][2], data[i+1][2]], clr, ls=ls, ax=ax)
		plot_line([data[i][1], data[i+1][1]], [data[i][2], data[i+1][2]], '.'+clr, ax=ax)

	return ax

def plot_word_probabilities(ids, words, entropy, xlim, ylim, clr, ax=None, linestyle=None):

	ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
	ax.set_xticks([])

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	for i in range(len(ids)):
		ax.bar(i, np.exp(-entropy[ids[i]]), color='w', edgecolor=clr, ls=linestyle)
		ax.annotate(words[i], xy=(i - len(words[i])/(2 * 6), np.exp(-entropy[ids[i]]) + 0.01), fontsize=10)

	return ax

def plot2Dembeddings(path1, path2, epoch1, epoch2, start, end, clr1='g', clr2='b', ls1=None, ls2=None):
	
	words1 = np.loadtxt(path1 + '/words_' + str(epoch1) + '.out')
	words2 = np.loadtxt(path2 + '/words_' + str(epoch2) + '.out')

	hiddens1 = np.loadtxt(path1 + '/hiddens_' + str(epoch1) + '.out')
	hiddens2 = np.loadtxt(path2 + '/hiddens_' + str(epoch2) + '.out')

	entropy1 = np.loadtxt(path1 + '/entropy_' + str(epoch1) + '.out')
	entropy2 = np.loadtxt(path2 + '/entropy_' + str(epoch2) + '.out')

	# load corpus
	corpus_path = path1 + '/corpus.*.data'
	corpus_path = [f for f in glob.glob(corpus_path)][0]
	corpus = torch.load(corpus_path)

	tokens = corpus.valid[start:end].numpy()
	sentence = [corpus.dictionary.idx2word[token] for token in tokens]

	plt.plot(corpus.frequencies.numpy(), np.linalg.norm(words2, axis=1), 'o')
	plt.show()


	# make figure
	fig = plt.figure(figsize=(16, 3.2))
	fig.suptitle('\"' + ' '.join(sentence) + '\"')

	# plot word_embeddings
	ax = plt.subplot(1,4,1)
	plot_all_word_embeddings(words1, [corpus.dictionary.idx2word[i] for i in range(words1.shape[0])], XLIM_WORDS, YLIM_WORDS, clr1, ax=ax)
	plot_all_word_embeddings(words2, [corpus.dictionary.idx2word[i] for i in range(words1.shape[0])], XLIM_WORDS, YLIM_WORDS, clr2, ax=ax)

	# plot hidden states
	ax = plt.subplot(1,4,2)
	ax = plot_sequence(hiddens1, start, end, XLIM_EMB, YLIM_EMB, clr=clr1, ls=ls1, ax=ax)
	ax = plot_sequence(hiddens2, start, end, XLIM_EMB, YLIM_EMB, clr=clr2, ls=ls2, ax=ax)


	ax = plt.subplot(1,4,3)
	xlim = [-1, len(tokens)]
	ax = plot_word_probabilities(tokens, sentence, entropy1, xlim, YLIM_HIST, 'g', ax=ax)

	ax = plt.subplot(1,4,4)
	xlim = [-1, len(tokens)]
	ax = plot_word_probabilities(tokens, sentence, entropy2, xlim, YLIM_HIST, 'b', ax=ax, linestyle=None)

	return fig

def rotate_sequence(sequence):
	 
	 # find mean and define goal vector
	 mean = np.mean(sequence, axis=0) / np.linalg.norm(np.mean(sequence, axis=0))
	 goal = np.array([0, 1])

	 # get rotation matrix
	 cos = np.dot(mean, goal)
	 theta1 = np.arccos(np.clip(cos, -1, 1))
	 theta2 = -theta1
	 
	 c1, s1 = np.cos(theta1), np.sin(theta1)
	 rot1 = np.array(((c1, -s1),(s1, c1)))

	 c2, s2 = np.cos(theta2), np.sin(theta2)
	 rot2 = np.array(((c2, -s2),(s2, c2)))

	 rotated1 = np.dot(rot1, sequence.transpose()).transpose()
	 rotated2 = np.dot(rot2, sequence.transpose()).transpose()

	 angle1 = np.dot(np.mean(rotated1, axis=0), goal) / np.linalg.norm(np.mean(rotated1, axis=0))
	 angle2 = np.dot(np.mean(rotated2, axis=0), goal) / np.linalg.norm(np.mean(rotated2, axis=0))

	 if angle1 > angle2:
	 	return rotated1
	 else:
	 	return rotated2

def plot_lip_trees(path):

	fig = plt.figure(figsize=(9, 2))

	for i,n in enumerate([2, 4, 8, 16]):
	
		# get sequence and add origin
		hiddens = np.loadtxt(path + str(n) + '/hiddens_500.out')
		seq = np.concatenate([np.zeros((1,2)), hiddens[:-1, 1:]])

		# rotate sequence such that mean direction is [0, 1]
		rotated = rotate_sequence(seq)

		# plot it
		ax = fig.add_subplot(1, 4, i+1)
		ax.set_xlim([-1.1, 1.1])
		ax.set_ylim([-1.1, 1.1])
		ax.set_xticks([-1, 0, 1])
		ax.set_yticks([-1, 0, 1])

		ax.plot(rotated[:,0], rotated[:,1], 'k', LineWidth=0.5)
		ax.plot(rotated[1:,0], rotated[1:,1], 'k.', markersize=3)
		ax.scatter(0, 0, facecolors='none', edgecolors='k')

	return fig


fig = plot_lip_trees('../../results/lip/lip')
plt.show()
#fig.savefig('../text/report_plots/rnn_lip.pdf', bbox_inches='tight')

'''
path1 = '../../results/lipschitz_tests/lipschitz_test1_rnn'
epoch1 = 496
path2 = '../../results/lipschitz_tests/lipschitz_test1_rnn_nobias'

epoch2 = 491

fig1 = plot2Dembeddings(path1, path2, epoch1, epoch2, 0, 4)
plt.show()
#fig2 = plot2Dembeddings(path1, path2, epoch1, epoch2, 8, 12)
#fig3 = plot2Dembeddings(path1, path2, epoch1, epoch2, 16, 20)
'''

