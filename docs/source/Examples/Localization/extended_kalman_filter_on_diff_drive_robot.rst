Extended Kalman Filter on differential driven robot (Python)
============================================================

Overview
--------

The Kalman filter is an optimal filtering method for linear systems. In this example we will extended the Kalman filter to systems governed by non-linear dynamics. The resulting filter is known as the Extended Kalman Filter or EKF. The discussion below is only meant to be descriptive and we will not go much into details. There are many excellent references around you can further enhance your understanding. This example is taken from the awesome `PtyhonRobotics <https://github.com/AtsushiSakai/PythonRobotics>`_ library. 

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
- :math:`f ~~ \text{is the non-linear function describing the dynamics of the system}`	
- :math:`h ~~ \text{is the non-linear function describing the measurements that is the modeling of the sensors we use}`
- :math:`\mathbf{w}_k \text{is the error vector associated with the control signal at step} k`
- :math:`\mathbf{v}_k \text{is the error vector associated with the measurement signal at step} k`

Furthermore, we have the following matrices
	
- :math:`\mathbf{P}_k ~~ \text{is the covariance matrix at step} ~~ k`
- :math:`\hat{\mathbf{P}}_k \text{is the predicted covariance matrix at step} k`
- :math:`\mathbf{Q}_k \text{is the covariance matrix associated with the control signal at step} k`
- :math:`\mathbf{R}_k \text{is the covariance matrix associated with the measurement signal at step} k`
- :math:`\mathbf{K}_k \text{is the gain matrix at step} k`		
- :math:`\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}}|_{\mathbf{x}=\mathbf{x}_{k-1}}`
- :math:`\mathbf{L}_k = \frac{\partial f}{\partial \mathbf{w}}|_{\mathbf{x}=\mathbf{x}_{k-1}}`
- :math:`\mathbf{H}_k = \frac{\partial h}{\partial \mathbf{x}}|_{\mathbf{x}=\mathbf{x}_{k-1}}`
- :math:`\mathbf{M}_k = \frac{\partial h}{\partial \mathbf{v}}|_{\mathbf{x}=\mathbf{x}_{k-1}}`


The prediction step is as follows

Predict step
---------------

.. math::

   \hat{\mathbf{x}}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_{k}, \mathbf{0})

   \hat{\mathbf{P}}_k = \mathbf{F}_{k-1}\mathbf{P}_{k-1}\mathbf{F}^{T}_{k-1} + \mathbf{L}_{k-1}\mathbf{Q}_{k-1}\mathbf{L}^{T}_{k-1}

The correction step is as follows

Correct step
------------

.. math::

  \hat{\mathbf{K}}_k = \hat{\mathbf{P}}_{k}\mathbf{H}_{k}(\mathbf{H}_{k}\hat{\mathbf{P}}_{k}\mathbf{H}_{k}^{T} + \mathbf{M}_{k}\mathbf{R}_{k}\mathbf{M}^{T}_{k})^{-1}
  
  \mathbf{x}_k = \hat{\mathbf{x}}_k + \mathbf{K}_k(\mathbf{z}_{k} - h(\hat{\mathbf{x}}_k,  \mathbf{0}))
  
  \mathbf{P}_k = (\mathbf{I} - \mathbf{K}_{k}\mathbf{H}^{T}_{k}) \hat{\mathbf{P}}_{k}

Code
----


.. code-block::

	import numpy as np
	import math
	import matplotlib.pyplot as plt
	from scipy.spatial.transform import Rotation as Rotation
	from src.filtering.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig
	from src.utils import INFO
	
.. code-block::

	DT = 0.1
	SIM_TIME = 50.0
	SHOW_ANIMATION = True
	V0 = 1.0 # m/sec

	#  Simulation parameter
	INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
	GPS_NOISE = np.diag([0.5, 0.5]) ** 2
	
The motion model is encapsulated in the class below

.. code-block::

	class MotionModel(object):

	    def __init__(self):
		pass

	    def __call__(self, x, u) -> np.array:
		F = np.array([[1.0, 0, 0, 0],
		              [0, 1.0, 0, 0],
		              [0, 0, 1.0, 0],
		              [0, 0, 0, 0]])

		B = np.array([[DT * math.cos(x[2, 0]), 0],
		              [DT * math.sin(x[2, 0]), 0],
		              [0.0, DT],
		              [1.0, 0.0]])

		return F @ x + B @ u

	    def jacobian_matrix(self, x, u):
		yaw = x[2, 0]
		v = u[0, 0]
		jacobian_mat = np.array([
		    [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
		    [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
		    [0.0, 0.0, 1.0, 0.0],
		    [0.0, 0.0, 0.0, 1.0]])

		return jacobian_mat
	
Similarly, we encapsilate the observation model in the class below	
.. code-block::

	class ObservationModel(object):
	    def __init__(self):
		pass

	    def __call__(self, x: np.array):
		H = np.array([
		    [1, 0, 0, 0],
		    [0, 1, 0, 0]
		])

		z = H @ x
		return z

	    def jacobian_matrix(self):
		jacobian = np.array([
		    [1, 0, 0, 0],
		    [0, 1, 0, 0]
		])
		return jacobian
		
.. code-block::

	def collect_input(time: float, noise: bool=True) -> np.array:
	    v = V0
	    yawrate = 0.0 #rad/s
	    u = np.array([[v], [yawrate]])

	    if noise:
		# add noise to input
		ud = u + INPUT_NOISE @ np.random.randn(2, 1)
		return ud

	    return u
	    
.. code-block::

	def sensor_readings(x):

	    z = ObservationModel()(x) + GPS_NOISE @ np.random.randn(2, 1)
	    return z
	    
.. code-block::

	def plot_covariance_ellipse(xEst, PEst):
	    Pxy = PEst[0:2, 0:2]
	    eigval, eigvec = np.linalg.eig(Pxy)

	    if eigval[0] >= eigval[1]:
		bigind = 0
		smallind = 1
	    else:
		bigind = 1
		smallind = 0

	    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
	    a = math.sqrt(eigval[bigind])
	    b = math.sqrt(eigval[smallind])
	    x = [a * math.cos(it) for it in t]
	    y = [b * math.sin(it) for it in t]
	    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
	    rot = Rotation.from_euler('z', angle).as_matrix()[0:2, 0:2]
	    fx = rot @ (np.array([x, y]))
	    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
	    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
	    plt.plot(px, py, "--r")	
	    
.. code-block::

	if __name__ == '__main__':

	    # State Vector [x y yaw v]'
	    x_init = np.zeros((4, 1))
	    x_true = np.zeros((4, 1))
	    x_dr = np.zeros((4, 1))

	    # history vectors to track performance
	    h_x_est = x_init
	    h_x_true = x_true
	    h_x_dr = x_true
	    hz = np.zeros((2, 1))

	    # Covariance for EKF simulation
	    Q = np.diag([
		0.1,  # variance of location on x-axis
		0.1,  # variance of location on y-axis
		np.deg2rad(1.0),  # variance of yaw angle
		1.0  # variance of velocity
	    ]) ** 2  # predict state covariance

	    # Observation x,y position covariance
	    R = np.diag([1.0, 1.0]) ** 2

	    motion_model = MotionModel()
	    obs_model = ObservationModel()

	    # initialize the filter to use
	    ekf_config = EKFConfig(q_matrix=Q, r_matrix=R, motion_model=motion_model,
		                   observation_model=obs_model, l_matrix=np.eye(4))
	    ekf = ExtendedKalmanFilter(config=ekf_config, init_state=x_init)

	    # run the simulation for as long is needed
	    time = 0.0
	    counter = 0
	    while SIM_TIME >= time:
		time += DT

		# somehow the robot will be given
		# an input. collect_input simulates
		# this

		x_true = motion_model(x_true, collect_input(time=time, noise=False))

		u = collect_input(time=time)
		x_dr = motion_model(x_dr, u)

		# the robot somehow gets access to
		# the sensor readings
		z = sensor_readings(x_true)

		print("======================================")
		print("{0} Input readings={1}".format(INFO, u))
		print("{0} Sensor readings={1}".format(INFO, z))
		estimated_state, Pest = ekf.estimate(u, z)

		print("{0} Estimated state={1}".format(INFO, estimated_state))
		print("======================================")

		# store data history
		h_x_est = np.hstack((h_x_est, estimated_state))
		hxDR = np.hstack((h_x_dr, h_x_dr))
		hxTrue = np.hstack((x_true, x_true))
		hz = np.hstack((hz, z))

		if SHOW_ANIMATION:
		    plt.cla()
		    # for stopping simulation with the esc key.
		    plt.gcf().canvas.mpl_connect('key_release_event',
		                                 lambda event: [exit(0) if event.key == 'escape' else None])
		    plt.plot(hz[0, :], hz[1, :], ".g")
		    plt.plot(hxTrue[0, :].flatten(),
		             hxTrue[1, :].flatten(), "-b")
		    plt.plot(hxDR[0, :].flatten(),
		             hxDR[1, :].flatten(), "-k")
		    plt.plot(h_x_est[0, :].flatten(),
		             h_x_est[1, :].flatten(), "-r")
		    plot_covariance_ellipse(h_x_est, Pest)
		    plt.axis("equal")
		    plt.grid(True)
		    plt.savefig("ekf_localization_" + str(counter) + ".png")
		    plt.pause(0.001)

		counter += 1	     
	 
Results
-------

