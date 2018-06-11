"""
=======================
 Sparse filtering demo
=======================

This demos the code in the accompanying ``sparseFiltering.py``, on the image data
used by Ngiam et al., which can be downloaded from
``http://cs.stanford.edu/~jngiam/data/patches.mat``.
"""

import numpy as np
import pylab
from scipy.io import loadmat
from sparseFiltering import *

def displayData(X, example_width = False, display_cols = False):
	"""
	Display 2D data in a nice grid
	==============================

	Displays 2D data stored in X in a nice grid.
	"""
	# compute rows, cols
	m,n = X.shape
	if not example_width:
		example_width = int(np.round(np.sqrt(n)))
	example_height = (n/example_width)
	# Compute number of items to display
	if not display_cols:
		display_cols = int(np.sqrt(m))
	display_rows = int(np.ceil(m/display_cols))
	pad = 1
	# Setup blank display
	display_array = -np.ones((pad+display_rows * (example_height+pad),
		pad+display_cols * (example_width+pad)))
	# Copy each example into a patch on the display array
	curr_ex = 0
	for j in range(display_rows):
		for i in range(display_cols):
			if curr_ex>=m:
				break
			# Copy the patch
			# Get the max value of the patch
			max_val = abs(X[curr_ex,:]).max()
			i_inds = example_width*[pad+j * (example_height+pad)+q for q in range(example_height)]
			j_inds = [pad+i * (example_width+pad)+q
						for q in range(example_width)
						for nn in range(example_height)]
			try:
				newData = (X[curr_ex,:].reshape((example_height,example_width)))/max_val
			except:
				print X[curr_ex,:].shape
				print (example_height,example_width)
				raise
			display_array[i_inds,j_inds] = newData.flatten()
			curr_ex+=1
		if curr_ex>=m:
			break
	# Display the image
	pylab.imshow(display_array,vmin=-1,vmax=1,interpolation='nearest',cmap=pylab.cm.gray)
	pylab.xticks([])
	pylab.yticks([])


def main():
	print "Loading data ..."
	# Loads variable `data` (size 256x50000)
	data = loadmat('patches.mat')['data']
	# Remove DC
	# data = data[:,:10000]
	data -= data.mean(axis=0)
	# Train layer 1
	print "Training layer 1 ..."
	L1_size = 256 # Increase this for more features
	L1 = sparseFiltering(L1_size, data)
	print "Layer 1 done."
	# Show Layer 1 bases
	displayData(L1)
	# Feed-forward layer 1
	data1 = feedForwardSF(L1,data)
	data1 -= data1.mean(axis=0)
	# Train layer 2
	print "Training layer 2 ..."
	L2_size = 256
	L2 = sparseFiltering(L2_size, data1)
	print "Layer 2 done."
	# Visualize layer 2
	pylab.figure()
	num_viz = 10
	offset = 1
	for i in range(num_viz):
		j = offset+i
		# Find the sign of the unit with the maximum absolute values
		sign = L2[j,:].flatten()[abs(L2[j,:]).argmax()]
		reverseSortedIndices = list(reversed((sign*L2[j,:]).argsort().tolist()))
		pylab.subplot(1, num_viz, i+1)
		displayData(L1[reverseSortedIndices[:10],:],False,1)
	pylab.show()

if __name__ == "__main__":
	main()