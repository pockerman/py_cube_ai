Installation
============

Currently, ``CubeAI`` has two distinct development ports;  `Python port <https://github.com/pockerman/py_cube_ai>`_ and `C++ port <https://github.com/pockerman/cubeai>`_. Below are instructions how to
install either or both. One of the future goals is to merge these two ports.

Python installation
------------------- 

``PyCubeAI``  depends on several packages. Specifically, the following packages are required:

- `NumPy <https://numpy.org/>`_
- `PyTorch <https://pytorch.org/>`_
- `Webots <https://cyberbotics.com/#cyberbotics>`_
- `Sphinx <https://www.sphinx-doc.org/en/master/>`_


.. code-block:: console

	pip install -r requirements.txt
	
Execute tests
-------------


C++ installation
----------------

The C++ port of ``CubeAI`` has the following dependencies

- CMake
- Python >= 3.8
- `boost C++ libraries <https://www.boost.org/>`_
- `PyTorch C++ bindings <https://pytorch.org/>`_
- `Blaze <https://bitbucket.org/blaze-lib/blaze/src/master/>`_ (version >= 3.8) check `here <https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation>`_ how to configure and install Blaze
- Blas library, e.g. OpenBLAS (required by Blaze)
- `rlenvs_from_cpp <https://github.com/pockerman/rlenvs_from_cpp>`_
- `nlohmann_json <https://github.com/nlohmann/json>`_


If you choose PyTorch with CUDA then ```cuDNN``` library is also required. This is a runtime library containing primitives for deep neural networks.

Furthermore, CubeAI has the following integrated dependencies

- `matplotlib-cpp <https://github.com/lava/matplotlib-cpp>`_
- `better-enums <https://github.com/aantron/better-enums>`_

Installation then follows the usual steps

.. code-block:: console

	 mkdir build && cd build
	 cmake ..
	 make install
	 
	 
If you are using ```rl_envs_from_cpp``` you need to export the path to the Python version you are using. For ecample:

.. code-block:: console
	
	export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.8/"
	
or 

.. code-block:: console

	export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.10/"


Depending on the values of the ```CMAKE_BUILD_TYPE```, the produced shared library will be installed in ```CMAKE_INSTALL_PREFIX/dbg/``` or ```CMAKE_INSTALL_PREFIX/opt/``` directories.

Issues with C++ installation
----------------------------

- Could not find boost. On a Ubuntu machine you can install the boost libraries as follows

```
sudo apt-get install libboost-dev
```


- ``pyconfig.h`` not found

In this case we may have to export the path to your Python library directory as shown above.

- Problems with Blaze includes

``cubeai`` is using Blaze-3.8. As of this version the ``FIND_PACKAGE( blaze )`` command does not populate ``BLAZE_INCLUDE_DIRS``  therefore you manually have to set the variable appropriately for your system. So edit the project's ``CMakeLists.txt`` file and populate appropriately the variable ``BLAZE_INCLUDE_DIRS``.

- Could NOT find BLAS 

The ```Blaze``` library depends on BLAS so it has to be installed. On a Ubuntu machine this can be doe as follows

```
sudo apt-get install libblas-dev liblapack-dev
```

Generate documentation
----------------------

You will need `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in order to generate the API documentation. Assuming that Sphinx is already installed
on your machine execute the following commands (see also `Sphinx tutorial <https://www.sphinx-doc.org/en/master/tutorial/index.html>`_). 

.. code-block:: console

	sphinx-quickstart docs
	sphinx-build -b html docs/source/ docs/build/html



