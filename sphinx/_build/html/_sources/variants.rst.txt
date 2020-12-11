Variants
========

The codebase allows you to choose the variant for tuning and experimenting with the framework.

Basic Seldonian
---------------
To begin with, we implemented vanilla Seldonian algorithm to classify the datapoint into 2 groups with difference in true positives as the fairness constraint.

To use this mode, you need to add CLI parameter `base` as:

.. code-block::

    python main.py base

Improvements to confidence interval
-----------------------------------
In the candidate selection process, we used Hoeffding inequality confidence interval as follows-

.. math::

    estimate \pm 2 \sqrt{\frac{ln(1/\delta)}{2 |D_{safety}|}}

Instead, this interval can be improved by using a separate values for - a.) error in candidate estimate and b.) confidence interval in safety set as follows-

.. math::

    estimate \pm \sqrt{\frac{ln(1/\delta)}{2 |D_{safety}|}} + \sqrt{\frac{ln(1/\delta)}{2 |D_{candidate}|}}

This will specifically be helpful in cases where the difference between the sizes of the 2 data splits is huge.

To use this mode, you need to add CLI parameter `mod` as:

.. code-block::

    python main.py mod

Improvement in bound propagation around constant values
-------------------------------------------------------
As constants have fixed value, there is no need to wrap a confidence interval around them. Thus, the 
:math:`\delta` value can directly go to other variable child and need not be split equally into half in case of 
binary operator when the other child is a constant. The figures below show naive and improved implementation 
of bound propagation in case of constant value of a node of the same tree respectively.

.. image:: images/const.png

To use this mode, you need to add CLI parameter `const` as:

.. code-block::

    python main.py const

Improvement in bound propagation from union bound
-------------------------------------------------
A user may defined the fairness constraint in such a way that a particular element appears multiple times 
in the same tree. Instead of treating all those entities as independent elements, we can combine all the 
elements together union bound and then use the final value of :math:`\delta`. This will theoretically improve the 
bound and give us better accuracy and more valid solutions. 

Example: Suppose we have A appearing 3 times with :math:`\delta/2`, :math:`\delta/4` and :math:`\delta/8`. We can simply take the

.. math:: 
    \delta_{sum} = 7\delta/8

and find the confidence interval using that :math:`\delta`. The figures below show the naive and improved implement using this functionality respectively.

.. image:: images/bound-no.png

.. image:: images/bound-yes.png

To use this mode, you need to add CLI parameter `bound` as:

.. code-block::

    python main.py bound

Combining all of the above optimizations
----------------------------------------
This can be done by using the `opt` mode as:

.. code-block::

    python main.py opt

Optimization with Lagrangian/KKT
--------------------------------
To use Lagrangian/KKT technique to optimise the objective function to get candidate solution, several additional modification are done:

- Objective function: The implementation to find the candidate solution and setting the value of the objective function (which is minimized) is changed to the following-

.. math::
    -fhat + (\mu * upperbound)

- Value of :math:`\mu` : We calculate the value of :math:`\mu` as

.. math::
    -\nabla f( \theta^{*})/ \nabla g_{i}( \theta^{*})

which must be positive to support the inequality of the fairness constraint and thus, in case the value is negative, then, we hard-code it to some positive value (say, 1).

- Change prediction to continuous function: Classification is essentially a step function (0/1 in case of binary classifier as in this case). Thus, instead of getting a label, we change the function to give the probability of getting a label instead of exact label value. This helps us find the derivative of the function easily. This change must be made by the user when he/she changes the predict function for their use-case.

- 2-player approach to solve KKT: One of the ways to solve KKT optimization problem is to use a 2-player approach where we fix a value of :math:`\mu` and then optimize the function w.r.t. :math:`\theta` and then , we fix :math:`\theta` and optimize the function w.r.t. :math:`\mu`. This goes on until we converge to some value or exceed a specific number of iterations.  Instead of doing a 2-player approach, to fasten the optimization process, we did one run of this by using a single value of :math:`\mu`, fetched from derivative of log-loss divided by derivative of fairness constraint with the initial :math:`\theta` values and optimizing the Lagrangian value using Powell optimizer.


