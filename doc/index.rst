.. gplearn documentation master file, created by
   sphinx-quickstart on Sun Apr 19 18:40:35 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gplearn's documentation!
===================================

|

.. image:: logos/gplearn-wide.png
    :align: center

|

.. math::
    One \,general \,law, \,leading \,to \,the \,advancement \,of \,all \,organic \,beings, namely,

.. math::
    multiply, \,vary, \,let \,the \,strongest \,live \,and \,the \,weakest \,die.

.. math::    
    - Charles \,Darwin, \,On \,the \,Origin \,of \,Species \,(1859)

|
|

.. currentmodule:: gplearn.genetic

`gplearn` implements Genetic Programming in Python, with a
`scikit-learn <http://scikit-learn.org>`_ inspired and compatible API.

While Genetic Programming (GP) can be used to perform a
`very wide variety of tasks <http://www.genetic-programming.org/combined.php>`_,
gplearn is purposefully constrained to solving symbolic regression problems.
This is motivated by the scikit-learn ethos, of having powerful estimators that
are straight-forward to implement.

Symbolic regression is a machine learning technique that aims to identify an
underlying mathematical expression that best describes a relationship. It
begins by building a population of naive random formulas to represent a
relationship between known independent variables and their dependent variable
targets in order to predict new data. Each successive generation of programs is
then evolved from the one that came before it by selecting the fittest
individuals from the population to undergo genetic operations.

gplearn retains the familiar scikit-learn `fit/predict` API and works with the
existing scikit-learn `pipeline <http://scikit-learn.org/stable/modules/pipeline.html>`_
and `grid search <http://scikit-learn.org/stable/modules/grid_search.html>`_
modules. The package attempts to squeeze a lot of functionality into a
scikit-learn-style API. While there are a lot of parameters to tweak, reading
the documentation here should make the more relevant ones clear for your
problem.

gplearn currently supports regression through the :class:`SymbolicRegressor` as
well as transformation for automated feature engineering with the
:class:`SymbolicTransformer`, which is designed to support regression problems,
but should also work for binary classification. Future versions of the package
will expand this class to support more complicated multi-target classification
problems, and much more is planned too.

gplearn is built on scikit-learn and a fairly recent copy (0.15.2+) is required
for installation. If you come across any issues in running or installing the
package, `please submit a bug report <https://github.com/trevorstephens/gplearn/issues>`_.

I hope you get some excellent results from using gplearn! If you do, please
`drop me a line on my blog <http://trevorstephens.com>`_ about how you used it.

Next up, read some more details about :ref:`what Genetic Programming is <intro>`,
and how it works...

Contents:

.. toctree::
   :maxdepth: 2

   intro
   examples
   reference
   advanced
   installation
   contributing
   changelog
