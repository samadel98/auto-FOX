#!/usr/bin/env python
"""Entry points for Auto-FOX.

Index
-----
.. currentmodule:: FOX.entry_points
.. autosummary::
    main_armc
    main_armc2yaml
    main_armc_report

API
---
.. autofunction:: main_armc
.. autofunction:: main_armc2yaml
.. autofunction:: main_armc_report

"""

import os
import argparse
from os.path import isfile, join
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import yaml
import pandas as pd

from ._nanoutils import UniqueLoader
from .armc import dict_to_armc, run_armc


def _parse_bool(value: str) -> bool:
    """Parse a command line boolean; ``bool("False")`` would be ``True``."""
    value_lower = value.lower()
    if value_lower in {'true', 'yes', '1'}:
        return True
    elif value_lower in {'false', 'no', '0'}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean such as 'True' or 'False'; got {value!r}")


def main_armc(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.classes.armc.run_armc`."""
    parser = argparse.ArgumentParser(
        prog='FOX',
        usage='init_armc filename -r restart',
        description=("Initialize the Auto-FOX Adaptive Rate Monte Carlo (ARMC) "
                     "parameter optimizer. "
                     "See 'https://auto-fox.readthedocs.io/en/latest/4_monte_carlo.html' for "
                     "a more detailed description.")
    )

    parser.add_argument(
        'filename', nargs=1, type=str, help='A .yaml file with ARMC settings.'
    )

    parser.add_argument(
        '-r', '--restart', nargs='?', type=_parse_bool, const=True, default=False,
        required=False, help='Whether or not to continue a previous calculation.'
    )

    args_parsed = parser.parse_args(args)
    filename: str = args_parsed.filename[0]
    restart: bool = args_parsed.restart

    with open(filename, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    armc, kwargs = dict_to_armc(dct)
    run_armc(armc, restart=restart, **kwargs)


def main_armc2yaml(args: Optional[list] = None) -> None:
    """Entrypoint for :meth:`FOX.classes.armc.ARMC.to_yaml`."""
    parser = argparse.ArgumentParser(
        prog='FOX',
        usage='armc2yaml filename -o output',
        description="Convert an ARMC .yaml file into a pre-processed .yaml file"
    )

    parser.add_argument(
        'filename', nargs=1, type=str, help='A .yaml file with ARMC settings.'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: the path+filename of the output .yaml file; '
              'will default to current working directory if not specified')
    )

    args_parsed = parser.parse_args(args)
    filename: str = args_parsed.filename[0]
    if not isfile(filename):
        raise FileNotFoundError(f"[Errno 2] No such file: {filename!r}")

    if args_parsed.output[0] is not None:
        output: str = args_parsed.output[0]
    else:  # Avoid duplicate names
        _output: str = join(os.getcwd(), 'armc.{:d}.yaml')
        i = 0
        while True:
            output = _output.format(i)
            if not isfile(output):
                break
            i += 1

    with open(filename, 'r') as f:
        dct_inp = yaml.load(f.read(), Loader=UniqueLoader)
    armc, kwargs = dict_to_armc(dct_inp)

    dct_out = armc.to_yaml_dict(
        path=kwargs['path'],
        folder=kwargs['folder'],
        logfile=kwargs['logfile'],
        psf=kwargs['psf'],
    )
    with open(output, 'w') as f:
        f.write(yaml.dump(dct_out))


def main_armc_report(args: Optional[list] = None) -> None:
    """Entrypoint for summarizing the results of an ARMC run.

    Prints the progress, acceptance rate, lowest error and best parameters of an ARMC .hdf5 file,
    writes the best parameters to ``best_param.csv`` and, if matplotlib is installed,
    plots the error per iteration and every PES descriptor of the best iteration
    against its reference.

    With more than one parameter set per iteration (ARMC-PT and CMA-ES)
    the best parameter set of the best iteration is reported.

    """
    parser = argparse.ArgumentParser(
        prog='FOX',
        usage='armc_report filename -o output',
        description="Summarize the results of an ARMC .hdf5 file",
    )

    parser.add_argument(
        'filename', type=str, help='The .hdf5 file of an ARMC run.'
    )

    parser.add_argument(
        '-o', '--output', type=str, metavar='output', default=None,
        help=("Optional: the directory for the .csv and .png files; "
              "defaults to a 'report' directory next to the .hdf5 file")
    )

    parser.add_argument(
        '--validation', action='store_true',
        help='Pick the best iteration with the validation errors rather than the normal errors.'
    )

    parser.add_argument(
        '--no-plots', action='store_true', help='Do not create any plots.'
    )

    args_parsed = parser.parse_args(args)
    filename = Path(args_parsed.filename)
    if not filename.is_file():
        raise FileNotFoundError(f"[Errno 2] No such file: {str(filename)!r}")
    if args_parsed.output is not None:
        output = Path(args_parsed.output)
    else:
        output = filename.parent / 'report'
    err_dset = 'validation/aux_error' if args_parsed.validation else 'aux_error'

    from .io import from_hdf5
    from .recipes import get_best
    from .recipes.param import _get_n_sets

    with h5py.File(filename, 'r') as f:
        super_iter_len, sub_iter_len = f['aux_error'].shape[:2]
        kappa = int(f.attrs['super-iteration'])
        omega = int(f.attrs['sub-iteration'])
        seed = f.attrs.get('seed')
        n_sets = _get_n_sets(f)
        pes_names = sorted(k[:-len('.ref')] for k in f.keys() if k.endswith('.ref'))
    if kappa < 0:
        raise ValueError(f"{str(filename)!r} does not contain any finished ARMC iterations")

    aux_error: pd.DataFrame = from_hdf5(filename, err_dset)
    acceptance: pd.DataFrame = from_hdf5(filename, 'acceptance')
    if n_sets > 1:
        # The total error of every parameter set, the PES descriptors being `f"{name}.{i}"`
        error = aux_error.T.groupby(lambda k: int(k.rpartition('.')[2])).sum(min_count=1).T
        j_best, i_best = np.unravel_index(int(np.nanargmin(error.values)), error.shape)
        j, i_set = int(j_best), int(i_best)
        lowest = error.iat[j, i_set]
        pes_names = [k for k in pes_names if k.rpartition('.')[2] == str(i_set)]
    else:
        error = aux_error.sum(axis=1, skipna=False).to_frame('error')
        j, i_set = int(error['error'].idxmin()), 0
        lowest = error.iat[j, 0]
    param: pd.Series = get_best(filename, 'param', i=i_set, err_dset=err_dset)

    print(f"ARMC results: {filename}")
    n_done = 1 + omega + kappa * sub_iter_len
    print(f"Finished iterations: {n_done} of {super_iter_len * sub_iter_len}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Acceptance rate: {100 * acceptance.values.mean():.1f}%")
    best_kappa, best_omega = divmod(j, sub_iter_len)
    location = f"super-iteration {best_kappa}, sub-iteration {best_omega}"
    if n_sets > 1:
        location += f", parameter set {i_set}"
    print(f"Lowest {'validation ' if args_parsed.validation else ''}error: "
          f"{lowest:.6g} ({location})")
    print("\nBest parameters:")
    print(param.to_string())

    output.mkdir(parents=True, exist_ok=True)
    written = [output / 'best_param.csv']
    param.to_csv(written[0])
    if not args_parsed.no_plots:
        # With more than one parameter set: the lowest error of every iteration
        plot_error = aux_error if n_sets == 1 else error.min(axis=1).to_frame('lowest error')
        written += _plot_armc_report(filename, output, plot_error, pes_names, err_dset)
    print("\nWritten: " + ", ".join(str(i) for i in written))


def _plot_armc_report(
    filename: Path,
    output: Path,
    aux_error: pd.DataFrame,
    pes_names: list[str],
    err_dset: str,
) -> list[Path]:
    """Plot the errors and PES descriptors for :func:`main_armc_report`."""
    try:
        import matplotlib
    except ImportError:
        print("\nmatplotlib is not installed; skipping the plots")
        return []
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .recipes import overlay_descriptor, plot_descriptor

    ret = [output / 'aux_error.png']
    ax = aux_error.plot(title=err_dset, xlabel='ARMC iteration', ylabel='Error')
    ax.figure.savefig(ret[0])
    plt.close(ax.figure)

    for pes_name in pes_names:
        name, _, i = pes_name.rpartition('.')
        dct = overlay_descriptor(filename, name, int(i), err_dset=err_dset)
        fig = plot_descriptor(dct, show_fig=False)
        ret.append(output / f'{pes_name}.png')
        fig.savefig(ret[-1])
        plt.close(fig)
    return ret
