.. _contributing:

Contributing
============

``gplearn`` welcomes your contributions! Whether it is a bug report, bug fix,
new feature or documentation enhancements, please help to improve the project!

In general, please follow the
`scikit-learn contribution guidelines <http://scikit-learn.org/stable/developers/contributing.html>`_
for how to contribute to an open-source project.

If you would like to open a bug report, please `open one here <https://github.com/trevorstephens/gplearn/issues>`_.
Please try to provide a `Short, Self Contained, Example <http://sscce.org/>`_
so that the root cause can be pinned down and corrected more easily.

If you would like to contribute a new feature or fix an existing bug, the basic
workflow to follow (as detailed more at the scikit-learn link above) is:

- `Open an issue <https://github.com/trevorstephens/gplearn/issues>`_ with what
  you would like to contribute to the project and its merits. Some features may
  be out of scope for ``gplearn``, so be sure to get the go-ahead before
  working on something that is outside of the project's goals.
- Fork the ``gplearn`` repository, clone it locally, and create your new feature
  branch.
- Make your code changes on the branch, commit them, and push to your fork.
- Open a pull request.

Please ensure that:

- Only data-dependent arguments should be passed to the fit/transform methods
  (``X``, ``y``, ``sample_weight``), and conversely, no data should be passed to the
  estimator initialization.
- No input validation occurs before fitting the estimator.
- Any new feature has great test coverage.
- Any new feature is well documented with
  `numpy-style docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
  & an example, if appropriate and illustrative.
- Any bug fix has regression tests.
- Comply with `PEP8 <https://pypi.python.org/pypi/pep8>`_.

Currently ``gplearn`` uses `Travis CI <https://travis-ci.org/trevorstephens/gplearn>`_
and `AppVeyor <https://ci.appveyor.com/project/trevorstephens/gplearn>`_
for testing, `Coveralls <https://coveralls.io/github/trevorstephens/gplearn>`_
for code coverage reports, and `Codacy <https://app.codacy.com/app/trevorstephens/gplearn>`_
for code quality checks. These applications should automatically run on your
new pull request to give you guidance on any problems in the new code.
