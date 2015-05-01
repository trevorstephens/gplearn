Introduction to GP
==================

|

.. image:: logos/gplearn-wide.png
    :align: center

|

.. math::
    Owing \,to \,this \,struggle \,for \,life,

.. math::
     any \,variation, \,however \,slight \,and \,from \,whatever \,cause \,proceeding,

.. math::
    if \,it \,be \,in \,any \,degree \,profitable \,to \,an \,individual \,of \,any \,species,

.. math::
    in \,its \,infinitely \,complex \,relations \,to \,other \,organic \,beings \,and \,to \,external \,nature,

.. math::
    will \,tend \,to \,the \,preservation \,of \,that \,individual,

.. math::
    and \,will \,generally \,be \,inherited \,by \,its \,offspring.

.. math::
    - \,Charles \,Darwin, \,On \,the \,Origin \,of \,Species \,(1859)

|
|

.. currentmodule:: gplearn.genetic

`gplearn` extends the `scikit-learn <http://scikit-learn.org>`_ machine learning library to perform Genetic Programming (GP) with symbolic regression.

Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression that best describes a relationship. It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets in order to predict new data. Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations.

Genetic programming is capable of taking a series of totally random programs, untrained and unaware of any given target function you might have had in mind, and making them breed, mutate and evolve their way towards the truth.

Think of genetic programming as a stochastic optimization process. Every time an initial population is conceived, and with every selection and evolution step in the process, random individuals from the current generation are selected to undergo random changes in order to enter the next. You can control this randomness by using the `random_state` parameter of the estimator.

So you're skeptical. Intrigued. Amazed perhaps. I hope so. Read on and discover the ways that the fittest programs in the population interact with one another to yield an even better generation.

