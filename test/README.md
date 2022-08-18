# Test `L3-solver` in various problems

## Setup
- `test_svm.py`: test SVMs.

- `test_unconstrain.py`: test (3) without constrains, that is, `A = 0` and `b = 0`

- `test_main.py` is for a general problem in (3)

## Reference methods

- `liblinear` - solve primal in (3) when SVMs 
- `CVXPY` - solve primal in (3)
- `CVXPY` - solve QP in (4)
- `ADMM` - solve QP in (4)

## Requirements
- pandas==1.4.2
- numpy==1.22.3
- cvxpy==1.2.1

## To-do

- [ ] Add our algorithm
