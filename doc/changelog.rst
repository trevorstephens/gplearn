.. currentmodule:: gplearn
.. _changelog:

Release History
===============

Version 0.2.0
-------------

- Allow users to define their own functions for use in genetic programs.
  Supported by the :func:`functions.make_function()` factory function. Using this
  a user may define any mathematical relationship with any number of arguments
  and grow totally customized programs. This also required modifying the API
  with the deprecation of the `comparison`, `transformer` and `trigonometric`
  arguments to the :class:`genetic.SymbolicRegressor` and
  :class:`genetic.SymbolicTransformer` classes in favor of the new
  `function_set` where any combination of preset and user-defined functions can
  be supplied. To restore previous behavior initialize the estimator with
  `function_set=['add2', 'sub2', 'mul2', 'div2', 'sqrt1', 'log1', 'abs1',
  'neg1', 'inv1', 'max2', 'min2']`.
- Reduce memory consumption for large datasets, large populations or many
  generations. Indices for in-sample/out-of-sample fitness calculations are now
  generated on demand rather than being stored in the program objects which
  reduces the size significantly for large datasets. Additionally "irrelevant"
  programs from earlier generations are removed if they did not contribute to
  the current population through genetic operations. This reduces the number of
  programs stored in the estimator which helps for large populations, high
  number of generations as well as for runs with significant bloat.


Version 0.1.0
-------------

Initial public release supporting symbolic regression tasks through the
:class:`genetic.SymbolicRegressor` class for regression problems and the
:class:`genetic.SymbolicTransformer` class for automated feature engineering.
