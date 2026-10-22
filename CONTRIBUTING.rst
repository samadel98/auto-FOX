############################
Contributing guidelines
############################

We welcome any kind of contribution to our software, from simple comment or question to a full fledged `pull request <https://help.github.com/articles/about-pull-requests/>`_. Please read and follow our `Code of Conduct <CODE_OF_CONDUCT.rst>`_.

A contribution can be one of the following cases:

1. You have a question.

2. You think you may have found a bug (including unexpected behavior).

3. You want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation).

The sections below outline the steps in each case.

You have a question
*******************

1. Use the search functionality `here <https://github.com/samadel98/auto-FOX/issues>`__ to see if someone already filed the same issue.

2. If your issue search did not yield any relevant results, make a new issue.

3. Apply the "Question" label; apply other labels when relevant.

You think you may have found a bug
**********************************

1. Use the search functionality `here <https://github.com/samadel98/auto-FOX/issues>`__ to see if someone already filed the same issue.

2. If your issue search did not yield any relevant results, make a new issue, making sure to provide enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:

   - The `SHA hashcode <https://help.github.com/articles/autolinked-references-and-urls/#commit-shas>`_ of the commit that is causing your problem;
   - Some identifying information (name and version number) for dependencies you're using;
   - Information about the operating system;

3. Apply relevant labels to the newly created issue.

You want to make some kind of change to the code base
*****************************************************

1. Announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue.

2. Wait until some kind of consensus is reached about your idea being a good idea.

3. If needed, fork the repository to your own Github profile and create your own feature branch off of the latest master commit. While working on your feature branch, make sure to stay up to date with the master branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions `here <https://help.github.com/articles/configuring-a-remote-for-a-fork/>`__ and `here <https://help.github.com/articles/syncing-a-fork/>`__).

4. Make sure the existing tests still work: install the package with ``pip install -e .[test]`` and run ``pytest``.
   The ARMC tests also need the test data package (``pip install https://github.com/nlesc-nano/auto-FOX-data/releases/download/1.2.0/Auto_FOX_Data-1.2.0-py3-none-any.whl``),
   and ``tests/test_armc.py::test_with_cp2k`` only runs when ``cp2k.ssmp`` is on your ``PATH``.
   Check the code style with ``pip install -e .[lint]``, ``flake8 FOX tests`` and ``pydocstyle FOX``.

5. Add your own tests (if necessary).

6. Update or expand the documentation, and check that it builds with ``pip install -e .[docs]`` and ``sphinx-build docs docs/_build/html``.

7. `Push <http://rogerdudler.github.io/git-guide/>`_ your feature branch to (your fork of) the Automated Forcefield Optimization Extension repository on GitHub.

8. Create the pull request, *e.g.* following the instructions `here <https://help.github.com/articles/creating-a-pull-request/>`__.

In case you feel like you've made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation: don't let this discourage you from making the pull request; we can help you! Just go ahead and submit the pull request, but keep in mind that you might be asked to append additional commits to your pull request.
