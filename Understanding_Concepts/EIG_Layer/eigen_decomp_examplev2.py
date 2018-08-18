import tensorflow as tf
import numpy as np
from numpy.linalg import eig
from numpy.linalg import pinv

#create upper triangular matrix
A_vec = np.array([-3,-2,4,-2,1,1,4,1,5])
A = np.reshape(A_vec,[3,3])
e = eig(A)
#extract eigenvalues and eigenvectors
e_val = e[0]
e_vec = e[1]

#tensorflow variables
A_tf = tf.Variable(tf.constant(A,dtype=tf.float64))
e_tf = tf.self_adjoint_eig(A_tf)
#extract eigenvalues and eigenvectors
e_val_tf = e_tf[0]
e_vec_tf = e_tf[1]

#initialise variables and run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#print A
print("A:")
print(A)

#show eigenvalues for Tensorflow and R
print("Tensorflow eigenvalues:",sess.run(e_val_tf))
print("Python eigenvalues:",e_val)


############# Eigenvector element gradient #############

eval_grad_tf = tf.gradients(e_val_tf[0], A_tf) #Tensorflow gradient of first eigenvalue w.r.t. A

#analytical gradient
eigenvec = e_vec[:,0]

#analytical deriv from analytical gradient of eigenvalue from Magnus (1985), "On differentiating
#Eigenvalues and Eigenvectors", Theorem 1 eqn (6)
eval_grad_analytical = np.outer(eigenvec,eigenvec)

print("Analytical gradient:")
print(eval_grad_analytical)
print("Tensorflow gradient:")
print(sess.run(eval_grad_tf))


############# Eigenvector element gradient #############

print("Tensorflow eigenvectors:")
print(sess.run(e_vec_tf))
print("Python eigenvectors:")
print(e_vec)

#tensorflow gradient
evec_grad_tf = tf.gradients(e_vec_tf[1,0], A_tf)

#analytical gradient

#analytical deriv from analytical gradient of eigenvalue from Magnus (1985), "On differentiating
#Eigenvalues and Eigenvectors", Theorem 1 eqn (7)

evec_grad_analytical = np.kron(eigenvec, pinv(e_val[0]*np.eye(3)-A)) #calculates derivatives for each component of eigenvector 3
evec_grad_analytical1 = np.reshape(evec_grad_analytical[1,:],[3,3]) #retrieves column 2 which should correspond to the gradient of eigenvector 3 element 2 w.r.t. A

print("Analytical gradient:")
print(evec_grad_analytical1)
print("Tensorflow gradient:")
print(sess.run(evec_grad_tf))
