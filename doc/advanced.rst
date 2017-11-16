.. _advanced:

Advanced Use
============

.. currentmodule:: gplearn

Saving Your Programs
--------------------

If you want to save your program for later use, you can use the ``pickle`` or
``cPickle`` libraries to achieve this::

    import cPickle as pickle

    est = SymbolicRegressor()
    est.fit(X_train, y_train)

Simply dump your model to a file::

    with open('gp_model.pkl', 'wb') as f:
        pickle.dump(est, f)

You can then load it at another date easily::

    with open('gp_model.pkl', 'rb') as f:
        est_gp = pickle.load(f)

And use it as if it was the Python session where you originally trained the
model.

Customizing Your Programs
-------------------------

This example demonstrates modifying the function set with your own user-defined
functions using the :func:`functions.make_function()` factory function.

First you need to define some function which will return a numpy array of the
correct shape. Most numpy operations will automatically do this. The factory
will perform some basic checks on your function to ensure it complies with
this. The function must also protect against zero division and invalid floating
point operations (such as the log of a negative number).

For this example we will implement a logical operation where two arguments are
compared, and if the first one is larger, return a third value, otherwise
return a fourth value::

    def logic(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)

To make this into a ``gplearn`` compatible function, we use the factory where
we must give it a name for display purposes and declare the arity of the
function which must match the number of arguments that your function expects::

    logical = make_function(function=logic,
                            name='logical',
                            arity=4)

This can then be added to a ``gplearn`` estimator like so::

    gp = SymbolicTransformer(function_set=['add', 'sub', 'mul', 'div', logical])

After fitting, you will see some of your programs will have used your own
customized functions, for example::

    sub(logical(X6, add(X11, 0.898), X10, X2), X5)

.. image:: images/ex3_fig1.png
    :align: center

Customizing Your Fitness Measure
--------------------------------

You can easily create your own fitness measure to have your programs evolve to
optimize whatever metric you need. This is done using the
:func:`fitness.make_fitness()` factory function. Let's say we wish to measure
our programs using MAPE (mean absolute percentage error). First we would need
to implement a function that returns this value. The function must take the
arguments ``y`` (the actual target values), ``y_pred`` (the predicted values
from the program) and ``w`` (the weights to apply to each sample) to work. For
MAPE, a possible solution is::

    def _mape(y, y_pred, w):
        """Calculate the mean absolute percentage error."""
        diffs = np.abs(np.divide((np.maximum(0.001, y) - np.maximum(0.001, y_pred)),
                                 np.maximum(0.001, y)))
        return 100. * np.average(diffs, weights=w)

Division by zero must be protected for a metric like MAPE as it is generally
used for cases where the target is positive and non-zero (like forecasting
demand). We need to keep in mind that the programs begin by being totally
naive, so a negative return value is possible. The ``np.maximum`` function will
protect against these cases, though you may wish to treat this differently
depending on your specific use case.

We then create a fitness measure for use in our evolution by using the
:func:`fitness.make_fitness()` factory function as follows::

    mape = make_fitness(_mape, greater_is_better=False)

This fitness measure can now be used to evolve a program that optimizes for
your specific needs by passing the new fitness object to the ``metric`` parameter
when creating an estimator::

    est = SymbolicRegressor(metric=mape, verbose=1)

.. currentmodule:: gplearn.genetic

Continuing Evolution With warm_start
------------------------------------

If you are evolving a lot of generations in your training session, but find
that you need to keep evolving more, you can use the ``warm_start`` parameter in
both :class:`SymbolicRegressor` and :class:`SymbolicTransformer` to continue
evolution beyond your original estimates. To do so, start evolution as usual::

    est = SymbolicRegressor(generations=10)
    est.fit(X, y)

If you then need to add further generations, simply change the ``generations``
and ``warm_start`` attributes and fit again::

    est.set_params(generations=20, warm_start=True)
    est.fit(X, y)

Evolution will then continue for a further 10 generations without losing the
programs that had been previously trained.