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
- `PyTorch C++ bindings <https://pytorch.org/>`_
- `Blaze <https://bitbucket.org/blaze-lib/blaze/src/master/>`_ (version >= 3.8)
- Blas library, e.g. OpenBLAS (required by Blaze)
- `gymfcpp <https://github.com/pockerman/gym_from_cpp>`_

Furthermore, CubeAI has the following integrated dependencies

- `matplotlib-cpp <https://github.com/lava/matplotlib-cpp>`_
- `better-enums <https://github.com/aantron/better-enums>`_

Installation then follows the usual steps

.. code-block:: console

	 ``mkdir build && cd build``
	 ``cmake ..``
	 ``make install``

Generate documentation
----------------------

You will need `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in order to generate the API documentation. Assuming that Sphinx is already installed
on your machine execute the following commands (see also `Sphinx tutorial <https://www.sphinx-doc.org/en/master/tutorial/index.html>`_). 

.. code-block:: console

	sphinx-quickstart docs
	sphinx-build -b html docs/source/ docs/build/html



