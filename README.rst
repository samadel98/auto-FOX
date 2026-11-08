
.. image:: https://github.com/samadel98/auto-FOX/actions/workflows/pythonpackage.yml/badge.svg?branch=master
    :target: https://github.com/samadel98/auto-FOX/actions/workflows/pythonpackage.yml?query=branch%3Amaster
.. image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg
    :target: https://samadel98.github.io/auto-FOX/
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3988142.svg
    :target: https://doi.org/10.5281/zenodo.3988142
.. image:: https://badge.fury.io/py/Auto-FOX.svg
    :target: https://badge.fury.io/py/Auto-FOX

|

.. image:: https://img.shields.io/badge/python-3.10-blue.svg
    :target: https://docs.python.org/3.10/
.. image:: https://img.shields.io/badge/python-3.11-blue.svg
    :target: https://docs.python.org/3.11/
.. image:: https://img.shields.io/badge/python-3.12-blue.svg
    :target: https://docs.python.org/3.12/
.. image:: https://img.shields.io/badge/python-3.13-blue.svg
    :target: https://docs.python.org/3.13/
.. image:: https://img.shields.io/badge/python-3.14-blue.svg
    :target: https://docs.python.org/3.14/

###########################################
Automated Forcefield Optimization Extension
###########################################

**Auto-FOX** is a library for analyzing potential energy surfaces (PESs) and
using the resulting PES descriptors for constructing forcefield parameters.
Further details are provided in the documentation_.


Currently implemented
=====================

This package is a work in progress; the following
functionalities are currently implemented:

- The MultiMolecule class, a class designed for handling and processing
  potential energy surfaces. (1_)
- A multi-XYZ reader. (2_)
- A radial and angular distribution generator (RDF & ADF). (3_)
- A root mean squared displacement generator (RMSD). (4_)
- A root mean squared fluctuation generator (RMSF). (5_)
- Tools for describing shell structures in, *e.g.*,
  nanocrystals or dissolved solutes. (6_)
- A Monte Carlo forcefield parameter optimizer (ARMC), including a
  parallel tempering variant (ARMC-PT). (7_)
- Readers and writers for CHARMM .psf, .prm and .rtf files and GROMACS .top
  files, with recipes for generating them. (psf_, prm_, rtf_, top_)
- Estimating Lennard-Jones parameters and computing non-bonded
  interactions. (lj_)
- Computing properties such as the pressure and bulk modulus
  from CP2K results. (properties_)

Using **Auto-FOX**
==================

- An input file with some basic examples is provided in
  the FOX.examples_ directory.

- An example MD trajectory of a CdSe quantum dot is included
  in the FOX.data_ directory.

  - The absolute path + filename of aforementioned trajectory
    can be retrieved as following:


.. code:: python

    >>> from FOX import example_xyz

- Further examples and more detailed descriptions are
  available in the documentation_.


Installation
============

Anaconda environments
---------------------

- While not strictly required, it is strongly recommended to use the
  virtual environments of Anaconda.

  - Available as either Miniconda_ or the complete Anaconda_ package.

- Anaconda comes with a built-in installer; more detailed installation
  instructions are available for a wide range of OSs.

  - See the `Anaconda documentation <https://docs.anaconda.com/anaconda/install/>`_.

- Anaconda environments can be created, enabled and disabled by,
  respectively, typing:

  - Create environment: ``conda create -n FOX -c conda-forge python pip``

  - Enable environment: ``conda activate FOX``

  - Disable environment: ``conda deactivate``

Installing **Auto-FOX**
-----------------------

- If using Conda, enable the environment: ``conda activate FOX``

- Install **Auto-FOX** from this repository:
  ``pip install git+https://github.com/samadel98/auto-FOX``

  - This also installs QMFlows from the `samadel98/qmflows <https://github.com/samadel98/qmflows>`_
    fork, which supports the current PLAMS versions, so ``git`` must be available.

  - ``pip install auto-FOX`` installs the last release on PyPI (0.10.2, from 2022),
    which predates the fixes in this repository (*e.g.* NumPy 2 and pandas 2 support).

- Congratulations, **Auto-FOX** is now installed and ready for use!

Optional dependencies
---------------------

- The plotting of data produced by **Auto-FOX** requires Matplotlib_.
  Matplotlib is distributed by both PyPi and Anaconda:

  - Anaconda:   ``conda install --name FOX -y -c conda-forge matplotlib``

  - PyPi:       ``pip install matplotlib``

- Construction of the angular distribution function in parallel requires DASK_.

  - Anaconda:   ``conda install --name FOX -y -c conda-forge dask``

- RDKit is required for a number of .psf-related recipes.

  - Anaconda:   ``conda install --name FOX -y -c conda-forge rdkit``

  - PyPi:       ``pip install rdkit``

.. _1: https://samadel98.github.io/auto-FOX/3_multimolecule.html
.. _2: https://samadel98.github.io/auto-FOX/5_xyz_reader.html
.. _3: https://samadel98.github.io/auto-FOX/1_rdf.html
.. _4: https://samadel98.github.io/auto-FOX/2_rmsd.html#root-mean-squared-displacement
.. _5: https://samadel98.github.io/auto-FOX/2_rmsd.html#root-mean-squared-fluctuation
.. _6: https://samadel98.github.io/auto-FOX/2_rmsd.html#discerning-shell-structures
.. _7: https://samadel98.github.io/auto-FOX/4_monte_carlo.html
.. _psf: https://samadel98.github.io/auto-FOX/8_psf.html
.. _prm: https://samadel98.github.io/auto-FOX/9_prm.html
.. _rtf: https://samadel98.github.io/auto-FOX/11_rtf.html
.. _top: https://samadel98.github.io/auto-FOX/12_top.html
.. _lj: https://samadel98.github.io/auto-FOX/6_estimate_lj.html
.. _properties: https://samadel98.github.io/auto-FOX/10_properties.html
.. _8: https://www.youtube.com/watch?v=hFDcoX7s6rE
.. _documentation: https://samadel98.github.io/auto-FOX/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Anaconda: https://www.anaconda.com/distribution/#download-section
.. _Matplotlib: https://matplotlib.org/
.. _FOX.data: https://github.com/samadel98/auto-FOX/blob/master/FOX/data
.. _FOX.examples: https://github.com/samadel98/auto-FOX/blob/master/FOX/examples/input.py
.. _DASK: https://dask.org/
.. _RDKit: https://www.rdkit.org/
