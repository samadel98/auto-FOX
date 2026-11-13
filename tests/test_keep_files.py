"""Tests for ``monte_carlo.keep_files`` and the clearing of job folders."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
import yaml
from schema import SchemaError
from assertionlib import assertion
from scm.plams import SingleJob, config, JobManager
from qmflows.utils import InitRestart

from FOX._nanoutils import UniqueLoader
from FOX.armc import dict_to_armc, PackageManager
from FOX.armc.package_manager import sync_job_names

PATH = Path('tests') / 'test_files'


class _Job(SingleJob):
    """A PLAMS job that is registered but never run."""

    def get_input(self) -> str:
        return ''

    def get_runscript(self) -> str:
        return ''


def _add_job(job_manager: JobManager, name: str, status: str = 'successful') -> _Job:
    """Register a job with **job_manager** and give it a folder and a status."""
    job = _Job(name=name)
    job_manager._register(job)  # PLAMS < 2025 creates the job folder here
    os.makedirs(job.path, exist_ok=True)
    job.status = status
    return job


def _job_folders(job_manager: JobManager) -> List[str]:
    return sorted(p.name for p in Path(job_manager.workdir).iterdir() if p.is_dir())


def _load_armc(keep_files: Any) -> Any:
    with open(PATH / 'armc_ref.yaml', 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    dct['monte_carlo']['keep_files'] = keep_files
    armc, _ = dict_to_armc(dct)
    return armc


@pytest.mark.parametrize('keep_files,ref', [
    (True, True),
    (False, False),
    (None, True),
    ('crashed', 'crashed'),
    ('Crashed', 'crashed'),
])
def test_keep_files_schema(keep_files: Any, ref: Any) -> None:
    """Test that ``keep_files`` accepts a boolean or ``"crashed"``."""
    armc = _load_armc(keep_files)
    assertion.eq(armc.keep_files, ref)
    assertion.eq(armc.to_yaml_dict()['monte_carlo']['keep_files'], ref)


@pytest.mark.parametrize('keep_files', ['yes', 'all', 1.5])
def test_keep_files_schema_raise(keep_files: Any) -> None:
    """Test that ``keep_files`` rejects other values."""
    assertion.assert_(_load_armc, keep_files, exception=SchemaError)


@pytest.mark.parametrize('keep_files,ref', [
    (True, []),
    (False, [{}]),
    ('crashed', [{'keep_crashed': True}]),
])
def test_keep_files_clear(keep_files: Any, ref: List[dict]) -> None:
    """Test which jobs :meth:`MonteCarloABC._clear_jobs` asks the package manager to clear."""
    armc = _load_armc(keep_files)
    calls: List[dict] = []
    armc.package_manager = SimpleNamespace(clear_jobs=lambda **kwargs: calls.append(kwargs))
    armc._clear_jobs()
    assertion.eq(calls, ref)


@pytest.mark.parametrize('keep_crashed', [False, True], ids=['all', 'crashed'])
def test_clear_jobs(tmp_path: Path, keep_crashed: bool) -> None:
    """Test :meth:`PackageManager.clear_jobs`."""
    with InitRestart(path=tmp_path, folder='workdir'):
        job_manager = config.default_jobmanager
        try:
            for status in ['successful', 'crashed', 'failed', 'successful']:
                _add_job(job_manager, 'md', status)
            assertion.eq(_job_folders(job_manager), ['md', 'md.002', 'md.003', 'md.004'])

            PackageManager.clear_jobs(keep_crashed=keep_crashed)
            assertion.eq(job_manager.jobs, [])
            assertion.eq(getattr(job_manager, '_job_full_name_map', {}), {})
            if keep_crashed:
                assertion.eq(_job_folders(job_manager), ['md.002', 'md.003'])
            else:
                assertion.eq(_job_folders(job_manager), [])

            # A new job must never get the name of a kept folder
            job = _add_job(job_manager, 'md')
            assertion.eq(job.name, 'md.005' if keep_crashed else 'md')
        finally:
            job_manager.jobs.clear()


def test_sync_job_names(tmp_path: Path) -> None:
    """Test that restarting in a folder with kept jobs never reuses their names."""
    workdir = tmp_path / 'workdir'
    for name in ['md', 'md.017', 'md.0', 'md.0.002', 'md.0.003']:
        (workdir / name).mkdir(parents=True)

    with InitRestart(path=tmp_path, folder='workdir'):
        job_manager = config.default_jobmanager
        try:
            sync_job_names(job_manager)
            assertion.eq(_add_job(job_manager, 'md').name, 'md.018')
            assertion.eq(_add_job(job_manager, 'md.0').name, 'md.0.004')
            assertion.eq(_add_job(job_manager, 'md.1').name, 'md.1')
        finally:
            job_manager.jobs.clear()
