Introduction
============

gplearn extends the `scikit-learn <http://scikit-learn.org>`_
machine learning library to perform genetic programming with symbolic
regression: 
A symbolic regressor is an estimator that begins by building a population
of naive random formulas to represent a relationship. The formulas are
represented as tree-like structures with mathematical functions being
recursively applied to variables and constants. Each successive generation
of programs is then evolved from the one that came before it by selecting
the fittest individuals from the population to undergo genetic operations
such as crossover, mutation or reproduction.

gplearn retains the familiar scikit-learn ``fit``/``predict`` API and works
with the existing scikit-learn pipeline and grid search modules.

