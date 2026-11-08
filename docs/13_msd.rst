.. _MSD:

Mean Squared Displacement & Diffusion
=====================================

Mean Squared Displacement
-------------------------
The mean squared displacement (MSD) measures how far atoms move, on average,
within a given lag time :math:`\tau`.
:meth:`FOX.MultiMolecule.init_msd` averages it over all atoms in an atom subset
and over all time origins of the trajectory,
so every frame contributes to every lag time:

.. math::

    \mathrm{MSD}(\tau) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M - \tau}
    \sum_{t=0}^{M - \tau - 1} \left( \mathbf{r}_{i}(t + \tau) - \mathbf{r}_{i}(t) \right)^2

This differs from the RMSD of :meth:`FOX.MultiMolecule.init_rmsd`,
which only measures displacements with respect to the first frame.
The MSD is calculated with fast Fourier transforms, which scale as
:math:`M \log M` rather than :math:`M^2` with the number of frames :math:`M`.

.. code:: python

    >>> from FOX import MultiMolecule, example_xyz

    >>> mol = MultiMolecule.from_xyz(example_xyz)
    >>> msd = mol.init_msd(timestep=1.0, atom_subset=('Cd', 'Se', 'O'))
    >>> msd.plot(title='MSD', xlabel='Time / fs', ylabel='MSD / Å$^2$')


.. plot::

    from FOX import MultiMolecule, example_xyz

    mol = MultiMolecule.from_xyz(example_xyz)
    msd = mol.init_msd(timestep=1.0, atom_subset=('Cd', 'Se', 'O'))
    msd.plot(title='MSD', xlabel='Time / fs', ylabel='MSD / Å$^2$')

The ``timestep`` is the time between two frames of the trajectory, in femtoseconds.
The ``atom_subset`` argument works as described in :ref:`RMSD`.


Periodic systems
----------------
Atoms that leave a periodic box re-enter it on the other side,
which would show up as jumps of a whole box length.
Pass ``periodic`` to unwrap the coordinates first, using the lattice vectors stored
in :attr:`FOX.MultiMolecule.lattice`:

.. code:: python

    >>> msd = mol.init_msd(timestep=1.0, atom_subset='O', periodic='xyz')

The unwrapping assumes that no atom moves more than half a box length between two frames.


Diffusion coefficients
----------------------
:meth:`FOX.MultiMolecule.init_diffusion_coefficient` fits a straight line to the MSD
and returns the self-diffusion coefficient :math:`D`, in m\ :sup:`2`/s,
from the Einstein relation:

.. math::

    D = \frac{1}{6} \frac{d\,\mathrm{MSD}(\tau)}{d\tau}

.. code:: python

    >>> mol.init_diffusion_coefficient(timestep=1.0, atom_subset=('Cd', 'Se', 'O'))
    Atoms
    Cd    1.624670e-10
    Se    1.667016e-10
    O     3.780643e-10
    Name: Diffusion coefficient / m^2 s^-1, dtype: float64

By default the fit uses lag times between 10% and 50% of the longest lag time.
This skips the ballistic regime at short times and the long times,
which are averaged over few time origins and are therefore noisy.
Use ``start`` and ``stop``, in fs, to choose another range,
ideally one where the MSD is a straight line.


As ARMC descriptors
-------------------
Both methods can be used as PES descriptors in ARMC (see :attr:`pes.block.func`).
Set ``max_lag`` (in frames) for the MSD, so that the reference and the MD trajectories
give MSDs of the same length, and make sure both trajectories store their frames
at the same time interval:

.. code:: yaml

    pes:
        msd:
            func: FOX.MultiMolecule.init_msd
            kwargs:
                atom_subset: [Cd, Se, O]
                max_lag: 50


API
---
.. automethod:: FOX.MultiMolecule.init_msd
    :noindex:

.. automethod:: FOX.MultiMolecule.init_diffusion_coefficient
    :noindex:

.. automodule:: FOX.functions.msd
