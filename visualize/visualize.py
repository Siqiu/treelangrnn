from dump import load_val_loss
import matplotlib.pyplot as plt

import numpy as np

def visualize_validation_loss(path, fig = None, ax=None, clr='k', label=''):

	if fig is None and ax is None:
		fig = plt.figure(figsize=(6, 6))
		ax = fig.add_subplot(1, 1, 1)

	val_loss = load_val_loss(path)
	epochs = len(val_loss)

	fig.suptitle('Validation Loss', fontsize=12)

	#y2 = [8.369777253157016, 8.131880289432543, 7.867192445524085, 7.808935784508116, 7.755466604740268, 7.567634225893693, 7.5915179140655695, 7.529823793575911, 7.656001874313179, 7.275944630775819,7.556750927716817,7.184588942891734]

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Cross Entropy')
	ax.set_ylim([0., 10.0])
	#ax.plot([i+1 for i in range(epochs)], val_loss, clr)
	ax.plot([i+1 for i in range(100)], [v for v in val_loss]+ [val_loss[-1]]*(100-len(val_loss)), clr, label=label)
	#ax.plot([i+1 for i in range(epochs)], y2 + ([7.184588942891734]*(epochs - len(y2))), 'k:')
	ax.legend()

	return fig, ax


#plt.axis([0, 500, , 0.85])
fig, ax = visualize_validation_loss('../../results/penn_100/val_loss.out.out', clr='k', label='sgd (lr=1e+0)')
fig, ax = visualize_validation_loss('../../results/penn_100_adam0001/val_loss.out.out', fig=fig, ax=ax, clr='k:', label='adam (lr=1e-4)')
fig, ax = visualize_validation_loss('../../results/penn_100_uniform/val_loss.out.out', fig=fig, ax=ax, clr='b', label='sgd, uni')
fig, ax = visualize_validation_loss('../../results/penn_50_uniform_adam00001/val_loss.out.out', fig=fig, ax=ax, clr='b:', label='adam uni')
fig ,ax = visualize_validation_loss('../../results/penn_vv_100/val_loss.out', fig=fig, ax=ax, clr='r', label='reference')
plt.show()
#fig.savefig('Validation Loss SGD ADAM.png')
