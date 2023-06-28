.. image:: crepes_logo.png

.. raw:: html

   <hr>

.. title:: crepes

``crepes`` is a Python package that implements conformal classifiers,
regressors, and predictive systems, on top of any standard classifier
and regressor, transforming the original predictions into
well-calibrated p-values and cumulative distribution functions, or
prediction sets and intervals with coverage guarantees.

The ``crepes`` package implements standard and Mondrian conformal
classifiers as well as standard, normalized and Mondrian conformal
regressors and predictive systems. While the package allows you to use
your own functions to compute difficulty estimates, non-conformity
scores and Mondrian categories, there is also a separate module,
called ``crepes.extras``, which provides some standard options for
these.

.. raw:: html

   <hr>
  
.. toctree::
    :maxdepth: 1
    
    Getting started <getting_started.rst>
    The crepes package <crepes>	      
    The crepes.extras module <crepes.extras>
    Examples <crepes_nb_wrap.ipynb>
    More examples <crepes_nb.ipynb>
    Citing crepes <citing.md>
    References <references.md>
