# TT-HJB
Tensor Train (TT) implementation of the Newton policy iteration for Hamilton-Jacobi-Bellman (HJB) equations. See [[Dolgov, Kalise, Kunisch, arXiv:1908.01533](https://arxiv.org/abs/1908.01533)] for the mathematical description.

## Installation
The code is based on [TT-Toolbox](https://github.com/oseledets/TT-Toolbox) and [tAMEn](https://github.com/dolgov/tamen) Matlab packages. Download or clone both repositories and add all subdirectories to the Matlab path.


## Contents
Detailed description of each file is provided in the beginning, and is also accessible through the Matlab `help` function. For example, see `help('hjb_leg')` for the syntax of the TT-HJB solver.


### Numerical test scripts
These are the top-level scripts that should be run to reproduce the numerical experiments from the paper.

* `test_hjb_allencahn1.m` 1-dimensional Allen-Cahn equation (Section 4.1). Control constraints can be turned on by setting a finite *umax* parameter.
* `test_hjb_allencahn2.m` 2-dimensional Allen-Cahn equation. *Note that the 2D test can take a lot of CPU time.*
* `test_hjb_fokker.m` Fokker-Planck equation (Section 4.2).
* `parse_parameter.m` Auxiliary file to input parameters

All tests ask the user to enter model and approximation parameters from the keyboard. Default parameters are provided in the hint and can be used as a starting experiment that should complete in a few minutes. The scripts populate the main Matlab workspace with all variables such as the TT format of the value function. See the description in the beginning of each file for a list of those.

### HJB solver

* `hjb_leg.m` The main computational routine to introduce the Legendre discretization, expand the system functions in this basis and run the policy iteration with TT approximations.
* `controlfun_leg.m` Computes the control signal at a particular state given the TT format of the value function.

### Discretization

* `legendre_rec.m` Computes Legendre polynomials and their derivatives
* `lgwt.m` [Legendre-Gauss Quadrature Weights and Nodes](https://uk.mathworks.com/matlabcentral/fileexchange/4540-legendre-gauss-quadrature-weights-and-nodes)
* `spectral_chebdiff.m` Computes the Chebyshev differentiation matrix

### Auxiliary

* `amen_cross_s.m` Enhanced TT-Cross algorithm for the TT approximation.
* `fp1d.m` Generates system matrices and vectors for the discretized Fokker-Planck equation
* `fpmr.m` Balanced Truncation of the Fokker-Planck model
* `quadratic_fun.m` Computes a multivariate quadratic polynomial for the LQR initial guess for the value function


