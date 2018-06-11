"""
==================
 Sparse filtering
==================

Python port of the sparse filtering matlab code by Jiquan Ngiam.

Requires numpy and scipy installed.
"""
import numpy as np
from scipy.optimize import minimize


def l2row(X):
	"""
	L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
	"""
	N = np.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N


def l2rowg(X,Y,N,D):
	"""
	Backpropagate through Normalization.

	Parameters
	----------

	X = Raw (possibly centered) data.
	Y = Row normalized data.
	N = Norms of rows.
	D = Deltas of previous layer. Used to compute gradient.

	Returns
	-------

	L2 normalized gradient.
	"""
	return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T


def sparseFiltering(N,X):
	"N = # features, X = input data (examples in column)"
	optW = np.random.randn(N,X.shape[0])

	# Objective function!
	def objFun(W):
		# Feed forward
		W = W.reshape((N,X.shape[0]))
		F = W.dot(X)
		Fs = np.sqrt(F**2 + 1e-8)
		NFs, L2Fs = l2row(Fs)
		Fhat, L2Fn = l2row(NFs.T)
		# Compute objective function
		# Backprop through each feedforward step
		DeltaW = l2rowg(NFs.T, Fhat, L2Fn, np.ones(Fhat.shape))
		DeltaW = l2rowg(Fs, NFs, L2Fs, DeltaW.T)
		DeltaW = (DeltaW*(F/Fs)).dot(X.T)
		return Fhat.sum(), DeltaW.flatten()

	# Actual optimization
	w,g = objFun(optW)
	res = minimize(objFun, optW, method='L-BFGS-B', jac = True, options = {'maxiter':200})
	return res.x.reshape(N,X.shape[0])


def feedForwardSF(W,X):
	"Feed-forward"
	F = W.dot(X)
	Fs = np.sqrt(F**2 + 1e-8)
	NFs = l2row(Fs)[0]
	return l2row(NFs.T)[0].T
