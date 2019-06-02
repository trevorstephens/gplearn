"""scikit-learn's estimator checks are written around multi-class estimators,
so SymbolicClassifier fails almost all the checks... Here we rewrite most of
that module with binary classification in mind."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import warnings
from copy import deepcopy
from functools import partial
from inspect import signature

import numpy as np

from sklearn.utils import _joblib
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import set_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import create_memmap_backed_data

from sklearn.base import clone

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.utils import shuffle
from sklearn.utils.validation import has_fit_parameter
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils.estimator_checks import (_apply_on_subsets,
                                            _yield_all_checks,
                                            choose_check_classifiers_labels,
                                            check_classifiers_predictions,
                                            check_no_attributes_set_in_init,
                                            check_parameters_default_constructible,
                                            is_public_parameter,
                                            multioutput_estimator_convert_y_2d,
                                            pairwise_estimator_convert_X)
from sklearn.utils.testing import SkipTest
from sklearn.exceptions import SkipTestWarning
from sklearn.utils import estimator_checks


def custom_check_estimator(Estimator):
    # Same as sklearn.check_estimator, skipping tests that can't succeed.
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()
        check_parameters_default_constructible(name, Estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    for check in _yield_all_checks(name, estimator):
        if (check is estimator_checks.check_estimators_dtypes or
                check is estimator_checks.check_fit_score_takes_y or
                check is estimator_checks.check_dtype_object or
                check is estimator_checks.check_sample_weights_list or
                check is estimator_checks.check_estimators_overwrite_params or
                check is estimator_checks.check_classifiers_classes or
                check is estimator_checks.check_supervised_y_2d or
                check is estimator_checks.check_fit2d_predict1d or
                check is estimator_checks.check_class_weight_classifiers or
                check is estimator_checks.check_methods_subset_invariance or
                check is estimator_checks.check_dont_overwrite_parameters or
                "check_estimators_fit_returns_self" in check.__repr__() or
                "check_classifiers_train" in check.__repr__()):
            continue
        try:
            check(name, estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


def rewritten_check_estimator(Estimator):
    # Same as sklearn.check_estimator, re-writing tests that can't succeed.
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()
        check_parameters_default_constructible(name, Estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    for check in _yield_rewritten_checks(name, estimator):
        try:
            check(name, estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


def _yield_rewritten_checks(name, estimator):
    yield check_estimators_dtypes
    yield check_sample_weights_list
    yield check_fit_score_takes_y
    yield check_dtype_object
    yield check_estimators_overwrite_params
    yield check_classifiers_classes
    yield check_supervised_y_2d
    yield check_fit2d_predict1d
    yield check_methods_subset_invariance
    yield check_dont_overwrite_parameters
    yield check_classifiers_train
    yield check_class_weight_classifiers
    yield partial(check_classifiers_train, readonly_memmap=True)
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)
    return None


def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = (X_train_int_64[:, 0] < 1).astype(int)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)


def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    if has_fit_parameter(estimator_orig, "sample_weight"):
        estimator = clone(estimator_orig)
        rnd = np.random.RandomState(0)
        X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                         estimator_orig)
        y = np.arange(10) % 2
        y = multioutput_estimator_convert_y_2d(estimator, y)
        sample_weight = [3] * 10
        # Test that estimators don't raise any exception
        estimator.fit(X, y, sample_weight=sample_weight)


def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rng.rand(40, 10), estimator_orig)
    X = X.astype(object)
    y = (X[:, 0] * 2).astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    try:
        estimator.fit(X, y.astype(object))
    except Exception as e:
        if "Unknown label type" not in str(e):
            raise

    X[0, 0] = {'foo': 'bar'}
    msg = "argument must be a string.* number"
    assert_raises_regex(TypeError, msg, estimator.fit, X, y)


def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = (X[:, 0] < 2).astype(np.int)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [key for key in dict_after_fit.keys()
                             if is_public_parameter(key)]

    attrs_added_by_fit = [key for key in public_keys_after_fit
                          if key not in dict_before_fit.keys()]

    # check that fit doesn't add any public attribute
    assert not attrs_added_by_fit, (
            'Estimator adds public attribute(s) during'
            ' the fit method.'
            ' Estimators are only allowed to add private attributes'
            ' either started with _ or ended'
            ' with _ but %s added'
            % ', '.join(attrs_added_by_fit))

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [key for key in public_keys_after_fit
                            if (dict_before_fit[key]
                                is not dict_after_fit[key])]

    assert not attrs_changed_by_fit, (
            'Estimator changes public attribute(s) during'
            ' the fit method. Estimators are only allowed'
            ' to change attributes started'
            ' or ended with _, but'
            ' %s changed'
            % ', '.join(attrs_changed_by_fit))


def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = (X[:, 0] > 1).astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(ValueError, "Reshape your data",
                                 getattr(estimator, method), X[0])


def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini bathes or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = (X[:, 0] > 1).astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "score_samples", "predict_proba"]:

        msg = ("{method} of {name} is not invariant when applied "
               "to a subset.").format(method=method, name=name)

        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X)
            assert_allclose(result_full, result_by_batch,
                            atol=1e-7, err_msg=msg)


def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = np.arange(10) % 2
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert args[1] in ["y", "Y"], (
                    "Expected y or Y as second argument for method "
                    "%s of %s. Got arguments: %r."
                    % (func_name, type(estimator).__name__, args))


def check_classifiers_train(name, classifier_orig, readonly_memmap=False):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if readonly_memmap:
        X_b, y_b = create_memmap_backed_data([X_b, y_b])

    for (X, y) in [(X_b, y_b)]:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, _ = X.shape
        classifier = clone(classifier_orig)
        X = pairwise_estimator_convert_X(X, classifier)
        y = multioutput_estimator_convert_y_2d(classifier, y)

        set_random_state(classifier)
        # raises error on malformed input for fit
        with assert_raises(
            ValueError,
            msg="The classifier {} does not "
                "raise an error when incorrect/malformed input "
                "data for fit is passed. The number of training "
                "examples is not the same as the number of labels. "
                "Perhaps use check_X_y in fit.".format(name)):
            classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        assert_greater(accuracy_score(y, y_pred), 0.83)

        # raises error on malformed input for predict
        msg = ("The classifier {} does not raise an error when the number of "
               "features in {} is different from the number of features in "
               "fit.")

        with assert_raises(ValueError,
                           msg=msg.format(name, "predict")):
            classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    assert_equal(decision.shape, (n_samples, 1))
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert_equal(decision.shape, (n_samples, n_classes))
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                with assert_raises(ValueError, msg=msg.format(
                        name, "decision_function")):
                    classifier.decision_function(X.T)
            except NotImplementedError:
                pass

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert_equal(y_prob.shape, (n_samples, n_classes))
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1),
                                      np.ones(n_samples))
            # raises error on malformed input for predict_proba
            with assert_raises(ValueError, msg=msg.format(
                    name, "predict_proba")):
                classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))


def check_estimators_fit_returns_self(name, estimator_orig,
                                      readonly_memmap=False):
    """Check if self is returned when calling fit"""
    X, y = make_blobs(random_state=0, n_samples=9, n_features=4)
    y = (y > 1).astype(int)
    # some want non-negative input
    X -= X.min()
    X = pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator


def check_supervised_y_2d(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)), estimator_orig)
    y = np.arange(10) % 2
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % (
        ", ".join([str(w_x) for w_x in w]))
    # check that we warned if we don't support multi-output
    assert_greater(len(w), 0, msg)
    assert "DataConversionWarning('A column-vector y" \
           " was passed when a 1d array was expected" in msg
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())


def check_classifiers_classes(name, classifier_orig):
    X_multiclass, y_multiclass = make_blobs(n_samples=30, random_state=0,
                                            cluster_std=0.1)
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass,
                                         random_state=7)
    X_multiclass = StandardScaler().fit_transform(X_multiclass)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X_multiclass -= X_multiclass.min() - .1

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_binary = pairwise_estimator_convert_X(X_binary, classifier_orig)

    labels_binary = ["one", "two"]

    y_names_binary = np.take(labels_binary, y_binary)

    for X, y, y_names in [(X_binary, y_binary, y_names_binary)]:
        for y_names_i in [y_names, y_names.astype('O')]:
            y_ = choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)


def check_estimators_overwrite_params(name, estimator_orig):
    X, y = make_blobs(random_state=0, n_samples=9)
    y = (y > 1).astype(int)
    # some want non-negative input
    X -= X.min()
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert_equal(_joblib.hash(new_value), _joblib.hash(original_value),
                     "Estimator %s should not change or mutate "
                     " the parameter %s from %s to %s during fit."
                     % (name, param_name, original_value, new_value))


def check_class_weight_classifiers(name, classifier_orig):
    # create a very noisy dataset
    X, y = make_blobs(centers=2, random_state=0, cluster_std=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)
    class_weight = {0: 1000, 1: 0.0001}
    classifier = clone(classifier_orig).set_params(
        class_weight=class_weight)
    set_random_state(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # XXX: Generally can use 0.89 here. On Windows, LinearSVC gets
    #      0.88 (Issue #9111)
    assert_greater(np.mean(y_pred == 0), 0.87)
