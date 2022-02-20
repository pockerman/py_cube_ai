Installation
============

``PyCubeAI``  depends on several packages. Specifically, the following packages are required:

- `NumPy <https://numpy.org/>`_
- `PyTorch <https://pytorch.org/>`_
- `Webots <https://cyberbotics.com/#cyberbotics>`_
- `Sphinx <https://www.sphinx-doc.org/en/master/>`_



.. code-block:: console

	pip install -r requirements.txt
	
Execute tests
-------------

Generate documentation
----------------------

You will need `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in order to generate the API documentation. Assuming that Sphinx is already installed
on your machine execute the following commands (see also `Sphinx tutorial <https://www.sphinx-doc.org/en/master/tutorial/index.html>`_). 

.. code-block:: console

	sphinx-quickstart docs
	sphinx-build -b html docs/source/ docs/build/html



