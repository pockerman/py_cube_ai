Some basic concepts (C++)
===============================

This example uses the ```DummyAlgorithm`` to train a ``DummyAgent`` agent. As its name
suggest, the ``DummyAgent`` is not really smart. However, this example illustrates some core
concepts in ``CubeAI``. Namely, we have three core ideas:

- A trainer class (see  `Trainer specification <../../Specs/trainer_specification.html>`_)
- An algorithm to train (see `Algorithm specification <../../Specs/algorithm_specification.html>`_)
- An agent that uses the output of the trained algorithm to step in the environemnt (see `Agent specification <../../Specs/agent_specification.html>`_)

Moreover, we use the `GymWorldWrapper <../../API/gym_world_wrapper.html>`_. This class wraps an OpenAI-Gym environment so that it conforms to 
the `DeepMind acme environment <https://github.com/deepmind/acme>`_.


Include files
-------------

Let's start with the necessary include files

.. code-block::

	#include "cubeai/base/cubeai_types.h"
	#include "cubeai/rl/algorithms/dummy/dummy_algorithm.h"
	#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
	#include "cubeai/rl/algorithms/rl_algo_config.h"

	#include "gymfcpp/gymfcpp_types.h"
	#include "gymfcpp/mountain_car_env.h"
	#include "gymfcpp/time_step.h"

	#include <iostream>
	
Further, we bring in scope some needed functionality

.. code-block:: 

	namespace example_0
	{

	using cubeai::real_t;
	using cubeai::uint_t;
	using cubeai::DynMat;
	using cubeai::DynVec;
	using cubeai::rl::algos::DummyAlgorithm;
	using cubeai::rl::algos::DummyAlgorithmConfig;
	using cubeai::rl::RLSerialAgentTrainer;
	using cubeai::rl::RLSerialTrainerConfig;
	using gymfcpp::MountainCar;

	}


Driver code
-----------

The main function is shown below. 

.. code-block::

	int main() {

	    using namespace example_0;

	    try{

		Py_Initialize();
		auto main_module = boost::python::import("__main__");
		auto gym_namespace = main_module.attr("__dict__");

		// create the environment
		MountainCar env("v0", gym_namespace, false);
		env.make();
		env.reset();

		// create the algorithm to train and configuration
		DummyAlgorithmConfig config = {100};
		DummyAlgorithm<MountainCar> algorithm(config);

		RLSerialTrainerConfig trainer_config = {100, 1000, 1.0e-8};

		RLSerialAgentTrainer<MountainCar, DummyAlgorithm<MountainCar>> trainer(trainer_config, algorithm);

		auto info = trainer.train(env);
		std::cout<<info<<std::endl;
	    }
	    catch(const boost::python::error_already_set&)
	    {
		    PyErr_Print();
	    }
	    catch(std::exception& e){
		std::cout<<e.what()<<std::endl;
	    }
	    catch(...){

		std::cout<<"Unknown exception occured"<<std::endl;
	    }

	   return 0;
	}

Results
-------

Running the code produces the following

.. code-block::

	Episode index........: 1
	Episode iterations...: 100
	Episode reward.......: -100
	Episode time.........: 0.0139128
	Has extra............: false

	Episode index........: 2
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 6.7e-08
	Has extra............: false

	Episode index........: 4
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.7e-08
	Has extra............: false

	Episode index........: 5
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.8e-08
	Has extra............: false

	Episode index........: 10
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.7e-08
	Has extra............: false

	Episode index........: 20
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.8e-08
	Has extra............: false

	Episode index........: 25
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.8e-08
	Has extra............: false

	Episode index........: 50
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.7e-08
	Has extra............: false

	Episode index........: 100
	Episode iterations...: 100
	Episode reward.......: 0
	Episode time.........: 1.8e-08
	Has extra............: false

	Converged...: false
	Tolerance...: 1e-08
	Residual....: 1.79769e+308
	Iterations..: 1000
	Total time..: 0.014297
