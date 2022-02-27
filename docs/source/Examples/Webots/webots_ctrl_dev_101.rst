Webots controller development-getting started
=============================================

In this example, we will some basic elements we need to take into account when developing controllers for the `Webots <https://cyberbotics.com/#cyberbotics>`_ simulator.
In fact, the tutorial herein is taken to a large extent from the official Webots documentation which can be found
in the `Controller Programming <https://cyberbotics.com/doc/guide/controller-programming?tab-language=python>`_ tutorial.

In this example, we will be using the `GCTronic e-puck <https://cyberbotics.com/doc/guide/epuck>`_ robot. We will go through the following items:

- How to start a controller
- Enabling and querying a sensor
- Send commands to an actuator
- Terminating the controller


We will assume that you already know how to create a simple world and add a robot node in it. If not, you can checkout 
`Tutorial 1: Your First Simulation in Webots (30 Minutes) <https://cyberbotics.com/doc/guide/tutorial-1-your-first-simulation-in-webots>`_. 

So let's get started. Open your project in Webots and add a controller in the world (Wizzards->New Robot Controller). Make sure that
you choose the Python API when you create the controller. This is the
controller we will exploit in this tutorial. We will create several main functions as we walk through this tutorial so that we
keep things as clean as possible. Creating the controller file via Webots results in the following code snippet

.. code-block::

	"""epuck_controller_dev_101 controller."""

	# You may need to import some classes of the controller module. Ex:
	#  from controller import Robot, Motor, DistanceSensor
	from controller import Robot

	# create the Robot instance.
	robot = Robot()

	# get the time step of the current world.
	timestep = int(robot.getBasicTimeStep())

	# You should insert a getDevice-like function in order to get the
	# instance of a device of the robot. Something like:
	#  motor = robot.getDevice('motorname')
	#  ds = robot.getDevice('dsname')
	#  ds.enable(timestep)

	# Main loop:
	# - perform simulation steps until Webots is stopping the controller
	while robot.step(timestep) != -1:
	    # Read the sensors:
	    # Enter here functions to read sensor data, like:
	    #  val = ds.getValue()

	    # Process sensor data here.

	    # Enter here functions to send actuator commands, like:
	    #  motor.setPosition(10.0)
	    pass

	# Enter here exit cleanup code.


Start a controller
------------------

The first function to create, is shown below. It simply, prints the name of the robot in the standard output

.. code-block::

	import math
	import numpy as np
	from controller import Robot
	
	TIME_STEP = 32


	def main_0(robot: Robot, limit: int = 4):

    		counter = 0
    		while robot.step(TIME_STEP) != -1:
        		print("Hello  from robot {0}".format(robot.getName()))
        		counter += 1
        		if counter >= limit:
            			break


	if __name__ == '__main__':

    		# create the Robot instance.
    		robot = Robot()

    		main_0(robot=robot)
		
The ``robot.step()`` function synchronizes the controller's data with the simulator. It must be present in every controller and it must be called at regular intervals.
Therefore it is usually placed in the main loop as in the above example. The value ``TIME_STEP=32`` specifies the duration of the control step. Thus, ``robot.step()`` wilrr compute 32 milliseconds of simulation and then return. You should note that this duration specifies amount of simulated time, not real (wall clock) time. Hence, it may actually take 1 millisecond or one minute of real time, depending on the complexity of the simulated world. 

``robot.step()`` will return -1 when Webots terminates the controller (see Controller Termination). Therefore, in this example, the control loop will run as long as the simulation runs. When the loop exists, no further communication with Webots is possible and the only option is to confirm to Webots to close the communication by calling the wb_robot_cleanup function.

Reading sensors
---------------

A robot understands the surrounding environment via sensors. Thus, we should be know how to read sensor values. 
The general pattern, is to use the ``robot.getDevice()`` function to access the sensor. Once we have access to the sensor, we can query it by using
the ``robot.getValue()`` function. Note however, that we need to explicitly enable the sensor. The following code snippet shows how to do this.

.. code-block::

	def main_1(robot: Robot, limit: int = 4):

    		sensor_ps0 = robot.getDevice("ps0")
    		sensor_ps0.enable(TIME_STEP)

    		counter = 0
    		while robot.step(TIME_STEP) != -1:
        		print("Hello  from robot {0}".format(robot.getName()))
        		print("Sensor value {0}".format(sensor_ps0.getValue()))
        		counter += 1
        		if counter >= limit:
            			break


	if __name__ == '__main__':

	    # create the Robot instance.
	    robot = Robot()

	    # main_0(robot=robot)
	    main_1(robot=robot)

The string passed to this function, "ps0" in this example, refers to a device name specified in the robot description (``.wbt`` or ``.proto``) file. If the robot has no device with the specified name, this function returns 0.

As we already mentioned, aach sensor must be enabled before it can be used. If a sensor is not enabled it returns undefined values. Enabling a sensor is achieved by using the corresponding ``sensor.enable()`` function. The input to this function is the update delay in milliseconds. The update delay specifies the desired interval between two updates of the sensor's data.
Most of the times, the update delay will be similar to the control step (``TIME_STEP``). Therefore, the sensor will be updated at every wb_robot_step function call.
However, this is not strictly necessary.  For example, the update delay is chosen to be twice the control step then the sensor data will be updated every two  function calls; this can be used to simulate a slow device. Moreover,  a larger update delay can also speed up the simulation, especially for CPU intensive devices like a camera. In contrast, it would be pointless to choose an update delay smaller than the control step, because it will not be possible for the controller to process the device's data at a higher frequency than that imposed by the control step. It is possible to disable a device at any time using the corresponding ``sensor.disable()`` function. This may increase the simulation speed.

In the usual case, the update delay is chosen to be similar to the control step (TIME_STEP) and hence the sensor will be updated at every wb_robot_step function call. If, for example, the update delay is chosen to be twice the control step then the sensor data will be updated every two wb_robot_step function calls: this can be used to simulate a slow device. Note that a larger update delay can also speed up the simulation, especially for CPU intensive devices like the Camera. On the contrary, it would be pointless to choose an update delay smaller than the control step, because it will not be possible for the controller to process the device's data at a higher frequency than that imposed by the control step. It is possible to disable a device at any time using the corresponding wb_*_disable function. This may increase the simulation speed.

The sensor value is updated during the call to the ``robot.step()`` function and  the call to the ``sensor.getValue()`` function retrieves the latest value.

Using actuators
---------------

The robot affects its surrounding environment via its actuators. Let's see how we can manipulate the actuators of the robot.
To start with, we need to fist access the actuator just like we did with the sensor above i.e. using the ``robot.getDevice()`` function and passing the
name of the device we want to access. We don't however, need to explicitly enable an actuator before using it. The following code snippet
shows how to make a rotational motor oscillate with a 2 Hz sine signal.


.. code-block::

	def main_2(robot: Robot, limit: int = 4) -> None:

	    left_motor = robot.getDevice("left wheel motor")
	    F = 2.0
	    t = 0.0
	    counter = 0
	    while robot.step(TIME_STEP) != -1:
		print("Hello  from robot {0}".format(robot.getName()))

		position = math.sin(t * 2.0 * np.pi * F)
		left_motor.setPosition(position)

		t += TIME_STEP / 1000.0
		counter += 1
		if counter >= limit:
		    break


	if __name__ == '__main__':

	    # create the Robot instance.
	    robot = Robot()

	    # main_0(robot=robot)
	    #main_1(robot=robot)
	    main_2(robot=robot)

In general, when we want to control a motion, we should try to decompose it into discrete steps that not surprisingly correspond to the control step.
Note that the ``motor.setPosition()`` function stores a new position request for the corresponding rotational motor but it does not immediately actuate the motor.
The effective actuation starts with the call to the ``robot.step()`` function. This function sends the actuation command to the motor but it does not wait for it to complete the motion (i.e. reach the specified target position); it just simulates the motor's motion for the specified number of milliseconds.


When the ``robot.step()`` function returns, the motor has moved by a certain (linear or rotational) amount which depends on several factors like the target position, the duration of the control step, the velocity, acceleration, force, and other parameters specified in the ``.wbt`` description of the motor. For example, if a very small control step or a low motor velocity is specified, the motor will not have moved much when the ``robot.step()`` function returns. In this case several control steps are required for the motor to reach the target position. If a longer duration or a higher velocity is specified, then the motor may have fully completed the motion when ``robot.step()`` returns.

As mentioned above, the ``motor.setPosition()`` function specifies only the desired target position. Just like with real robots, it is possible (in physics-based simulations only), that the motor is not able to reach this position, because it is blocked by obstacles or because the motor's torque (maxForce) is insufficient to oppose gravity, etc.

Terminate a controller
----------------------

When using the Python API, we don't need to do anything specifically to terminate a controller. Usually a controller process runs in an endless loop until it is terminated by Webots on one of the following events:

- Webots quits,
- The simulation is reset,
- The world is reloaded,
- A new simulation is loaded, or
- The controller name is changed (by the user from the scene tree GUI or by a supervisor process).

There are a few points we need to be aware of though. To start with, a controller cannot prevent its own termination. When one of the above events happens, the ``robot.step()`` function returns -1. From this point, Webots will not communicate with the controller any more. Therefore, new print statements executed by the controller on stdout or stderr will no longer appear in the Webots console. After one second (real time), if the controller has not terminated by itself, Webots will kill it (``SIGKILL``). That leaves a limited amount of time to the controller to save important data, close files, etc. before it is actually killed by Webots. 
