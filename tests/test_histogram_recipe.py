"""A module for testing :mod:`FOX.recipes.histogram`."""

from itertools import combinations_with_replacement

import pytest
import numpy as np
import pandas as pd
from assertionlib import assertion

from FOX.recipes.histogram import filter_atoms

COLUMNS = pd.MultiIndex.from_tuples(
    combinations_with_replacement(sorted(['Cd', 'Se', 'O', 'C']), r=2)
)
DF = pd.DataFrame(True, index=[0], columns=COLUMNS)


@pytest.mark.parametrize("atoms,filter_func,ref", [
    ({'Cd', 'Se'}, 'within_set', [('Cd', 'Cd'), ('Cd', 'Se'), ('Se', 'Se')]),
    (['Cd', 'Se'], 'WITHIN_SET', [('Cd', 'Cd'), ('Cd', 'Se'), ('Se', 'Se')]),
    ('Cd', 'within_set', [('Cd', 'Cd')]),
    (np.str_('Cd'), 'within_set', [('Cd', 'Cd')]),
    ('Cd', 'between_sets', [('C', 'Cd'), ('Cd', 'O'), ('Cd', 'Se')]),
    ({'Cd', 'Se'}, 'between_sets', [('C', 'Cd'), ('C', 'Se'), ('Cd', 'O'), ('O', 'Se')]),
    ({'Cd', 'Se'}, lambda atom_set, pair: 'O' in pair,
     [('C', 'O'), ('Cd', 'O'), ('O', 'O'), ('O', 'Se')]),
], ids=["set", "list", "str", "np_str", "between_sets", "between_sets2", "callable"])
def test_filter_atoms(atoms, filter_func, ref) -> None:
    """Test :func:`FOX.recipes.histogram.filter_atoms`."""
    df = DF.copy()
    out = filter_atoms(df, atoms, filter_func=filter_func)
    assertion.eq(list(out.columns), ref)
    assertion.eq(list(df.columns), list(COLUMNS))


@pytest.mark.parametrize("filter_func,exc", [
    ('bob', ValueError),
    (1, TypeError),
], ids=["value", "type"])
def test_filter_atoms_raise(filter_func, exc) -> None:
    """Test :func:`FOX.recipes.histogram.filter_atoms` with invalid **filter_func** values."""
    with pytest.raises(exc):
        filter_atoms(DF, 'Cd', filter_func=filter_func)
