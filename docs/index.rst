crepes
======

``crepes`` is a Python package for generating conformal regressors, which transform point predictions of any underlying regression model into prediction intervals for specified levels of confidence. The package also implements conformal predictive systems, which transform the point predictions into cumulative distribution functions.

The ``crepes`` package implements standard, normalized and Mondrian conformal regressors and predictive systems. While the package allows you to use your own difficulty estimates and Mondrian categories, there is also a separate module, called ``crepes.fillings``, which provides some standard options for these.

.. raw:: html

   <hr>
  
.. toctree::
    :maxdepth: 1
    
    Getting started <getting_started.rst>
    Examples <crepes_nb.ipynb>
    The crepes package <crepes>	      
    The crepes.fillings module <crepes.fillings>
    Citing crepes <citing.md>
    References <references.md>
