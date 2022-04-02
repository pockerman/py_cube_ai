Extended Kalman Filter on differential driven robot (Python)
============================================================

Overview
--------

The Kalman filter is an optimal filtering method for linear systems. In this example we will extended the Kalman filter to systems governed by non-linear dynamics. The resulting filter is known as the Extended Kalman Filter or EKF. The discussion below is only meant to be descriptive and we will not go much into details. There are many excellent references around you can further enhance your understanding. 

Extended Kalman filter
----------------------

Briefly, the EKF is an improvement over the classic Kalman Filter that can be applied to non-linear systems. The crux of the algorithm  remains the predictor-corrector steps just like in the Kalman Filter. In fact to a large extent the two methods are identical. However, the EKF method involves a linearization step of the non-linear system. 

Thus, in EKF, we linearize the nonlinear system around
the Kalman filter estimate, and the Kalman filter estimate is based on the
linearized system. This is the idea of the extended Kalman filter (EKF), which
was originally proposed by Stanley Schmidt so that the Kalman filter could be
applied to nonlinear spacecraft navigation problems.


- :math:`\mathbf{x}_k ~~ \text{is the state vector at step} ~~ k`

- :math:`\hat{\mathbf{x}}_k ~~ \text{is the predicted state vector at step} ~~ k`

- :math:`\mathbf{u}_k ~~ \text{is the control signal vector at step} ~~ k`
	
- :math:`\mathbf{z}_k ~~ \text{is the sensor measurement vector at step} ~~ k`
	
.. math::

	f ~~ \text{is the non-linear function describing the dynamics of the system}
	
.. math::
	
	h ~~ \text{is the non-linear function describing the measurements that is the modeling of the sensors we use}
	
.. math::
	
	\mathbf{P}_k ~~ \text{is the covariance matrix at step} ~~ k
		\item \hat{\mathbf{P}}_k \text{is the predicted covariance matrix at step} k
		\item \mathbf{Q}_k \text{is the covariance matrix associated with the control signal at step} k
		\item \mathbf{R}_k \text{is the covariance matrix associated with the measurement signal at step} k
		\item \mathbf{K}_k \text{is the gain matrix at step} k
		\item \mathbf{w}_k \text{is the error vector associated with the control signal at step} k
		\item \mathbf{v}_k \text{is the error vector associated with the measurement signal at step} k
		
$$\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}}|_{\mathbf{x}=\mathbf{x}_{k-1}}$$

$$\mathbf{L}_k = \frac{\partial f}{\partial \mathbf{w}}|_{\mathbf{x}=\mathbf{x}_{k-1}}$$

$$\mathbf{H}_k = \frac{\partial h}{\partial \mathbf{x}}|_{\mathbf{x}=\mathbf{x}_{k-1}}$$

$$\mathbf{M}_k = \frac{\partial h}{\partial \mathbf{v}}|_{\mathbf{x}=\mathbf{x}_{k-1}}$$


- Predict step

$$\hat{\mathbf{x}}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_{k}, \mathbf{0})$$

$$\hat{\mathbf{P}}_k = \mathbf{F}_{k-1}\mathbf{P}_{k-1}\mathbf{F}^{T}_{k-1} + \mathbf{L}_{k-1}\mathbf{Q}_{k-1}\mathbf{L}^{T}_{k-1}$$

- Correct step

Code
----
