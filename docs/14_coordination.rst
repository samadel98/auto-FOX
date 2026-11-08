.. _Coordination:

Coordination Numbers
====================
The coordination number of an atom is the number of neighbouring atoms within a
given distance, :math:`r_{\mathrm{max}}`.
For example, counting the O atoms around each Cd atom of a quantum dot
tells which Cd atoms sit at the surface and how many ligand O atoms bind to each of them.
Three :class:`FOX.MultiMolecule` methods count them:

* :meth:`~FOX.MultiMolecule.get_coordination_number`: the coordination number of every
  central atom in every molecule (frame).
* :meth:`~FOX.MultiMolecule.init_coordination_number`: the average coordination number
  per molecule, as a function of the molecular index.
* :meth:`~FOX.MultiMolecule.init_coordination_distribution`: the fraction of the central atoms
  with 0, 1, 2, ... neighbours, averaged over all molecules.

Atom pairs are given as ``(central atom, neighbour)``, so ``("Cd", "O")`` counts the
O atoms around each Cd atom. An atom is never counted as its own neighbour.

.. code:: python

    >>> from FOX import MultiMolecule, example_xyz

    >>> mol = MultiMolecule.from_xyz(example_xyz)
    >>> mol.init_coordination_distribution([("Cd", "O"), ("Cd", "Se")], r_max=3.0)
    Atom pairs               Cd O     Cd Se
    Coordination number
    0                    0.441176  0.000000
    1                    0.068466  0.003250
    2                    0.457199  0.527184
    3                    0.033159  0.105909
    4                    0.000000  0.363657

.. plot::

    from FOX import MultiMolecule, example_xyz

    mol = MultiMolecule.from_xyz(example_xyz)
    df = mol.init_coordination_distribution([("Cd", "O"), ("Cd", "Se")], r_max=3.0)
    df.plot.bar(title='Coordination of Cd', ylabel='Fraction of Cd atoms', rot=0)

Use a mapping for ``r_max`` to set a different distance per atom pair,
for example ``r_max={"Cd O": 2.6, "Cd Se": 3.0}``.
A good distance is usually the first minimum of the corresponding
radial distribution function (see :ref:`RDF`).
Pass ``periodic`` to count neighbours across the boundaries of a periodic box,
using the lattice vectors in :attr:`FOX.MultiMolecule.lattice`.


As ARMC descriptors
-------------------
:meth:`~FOX.MultiMolecule.init_coordination_distribution` can be used as a PES descriptor
in ARMC (see :attr:`pes.block.func`), comparing the coordination of the MD trajectories
with that of the reference. Set ``cn_max`` so that both have the same number of rows:

.. code:: yaml

    pes:
        coordination:
            func: FOX.MultiMolecule.init_coordination_distribution
            kwargs:
                atom_pairs: [[Cd, O], [Cd, Se]]
                r_max: 3.0
                cn_max: 4


API
---
.. automethod:: FOX.MultiMolecule.get_coordination_number
    :noindex:

.. automethod:: FOX.MultiMolecule.init_coordination_number
    :noindex:

.. automethod:: FOX.MultiMolecule.init_coordination_distribution
    :noindex:
