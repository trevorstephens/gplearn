#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"

if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s -v --with-coverage --cover-package=gplearn
else
    nosetests -s -v gplearn
fi

if [[ "$QUALITY" == "true" ]]; then
    prospector --doc-warnings gplearn || true;
fi

#make test-doc test-sphinxext
