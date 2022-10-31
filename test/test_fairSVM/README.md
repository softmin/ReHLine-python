# ReHLine: Test **FairSVM**

- `DCCP`: is FairSVM solver used in the [original paper](http://proceedings.mlr.press/v54/zafar17a/zafar17a-supp.pdf).

## CASE 1: Suboptimal results

- Results

        DCCP obj: 3965.25241 time: 8.66
        ReHLine obj: 4055.53691 time: 50.61

- Steps

        Iter 233140, dual_objfn = -3824.8, xi_diff = 0, beta_diff = 3.50764e-06
        Iter 233150, dual_objfn = -3824.81, xi_diff = 0, beta_diff = 1.25855e-06
        Iter 233160, dual_objfn = -3824.82, xi_diff = 0, beta_diff = 4.62956e-07
        Iter 233170, dual_objfn = -3824.82, xi_diff = 0, beta_diff = 1.73071e-07
        Iter 233180, dual_objfn = -3824.83, xi_diff = 0, beta_diff = 6.53014e-08
        Iter 233190, dual_objfn = -3824.84, xi_diff = 0, beta_diff = 2.47365e-08
        Iter 233200, dual_objfn = -3824.85, xi_diff = 0, beta_diff = 9.37288e-09
        Iter 233210, dual_objfn = -3824.86, xi_diff = 0, beta_diff = 3.54428e-09
        Iter 233220, dual_objfn = -3824.86, xi_diff = 0, beta_diff = 1.33592e-09

**RK**: It seems that `beta_diff` is convergent, yet the dual objective function is still decreasing. Is that because of the ill-posed QP?

- Data: `fairSVM_suboptimal.npz`
```python
np.savez('fairSVM_suboptimal', X=X, y=y, U=cue.U, V=cue.V, A=cue.A, b=cue.b)
```

## CASE 2: Slow convergence

- Results

        ReHLine obj: 482.08505 time: 86.80

- Steps
 
        Iter 374730, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.02037e-16
        Iter 374740, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.62844e-16
        Iter 374750, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.39731e-16
        Iter 374760, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.03921e-16
        Iter 374770, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.2508e-16
        Iter 374780, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.45568e-16
        Iter 374790, dual_objfn = -482.085, xi_diff = 3.02076e-07, beta_diff = 1.55128e-16
        Iter 374800, dual_objfn = -482.085, xi_diff = 3.24476e-07, beta_diff = 8.61147e-07
        Iter 374810, dual_objfn = -482.085, xi_diff = 3.12629e-07, beta_diff = 7.12207e-07
        Iter 374820, dual_objfn = -482.085, xi_diff = 3.0318e-07, beta_diff = 6.40738e-07
        Iter 374830, dual_objfn = -482.085, xi_diff = 2.96811e-07, beta_diff = 6.00748e-07
        Iter 374840, dual_objfn = -482.085, xi_diff = 2.93016e-07, beta_diff = 5.74499e-07
        Iter 374850, dual_objfn = -482.085, xi_diff = 2.90972e-07, beta_diff = 5.54806e-07
        Iter 374860, dual_objfn = -482.085, xi_diff = 2.8995e-07, beta_diff = 5.38682e-07
        Iter 374870, dual_objfn = -482.085, xi_diff = 2.89447e-07, beta_diff = 5.24817e-07

**RK**: Huge change from **Iter 374790** to **Iter 374800**, may need to figure it out...

- Data: `fairSVM_slow.npz`
```python
np.savez('fairSVM_slow', X=X, y=y, U=cue.U, V=cue.V, A=cue.A, b=cue.b)
```
