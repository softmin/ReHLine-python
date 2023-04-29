library(rehline)
library(hqreg)

# Simulate data
set.seed(123)
n = 5000
d = 100
kappa = runif(1)
x = matrix(rnorm(n * d), n, d)
beta0 = rnorm(d)
y = c(x %*% beta0) + rnorm(n, sd = 0.1)

lam1 = 0.05
lam2 = 0.05

# In our setting we penalize the whole beta vector
# The last element of beta is the intercept
objfn = function(beta, x, y, kappa, lam1, lam2)
{
    d1 = length(beta)
    pred = beta[d1] + c(x %*% beta[-d1])
    loss = kappa * pmax(y - pred, 0) + (1 - kappa) * pmax(pred - y, 0)
    reg = lam1 * sum(abs(beta)) + lam2 * sum(beta^2) / 2
    mean(loss) + reg
}

# hqreg does not penalize intercept, so we does not use the built-in
# intercept term, but manually add a one vector to x
#
# Reparameterization of penalty parameters
# lam1 = lambda * alpha
# lam2 = lambda * (1 - alpha)
# => alpha = lam1 / (lam1 + lam2), lambda = lam1 + lam2
xone = cbind(x, 1)
res1 = hqreg_raw(xone, y, method = "quantile", tau = kappa,
                 alpha = lam1 / (lam1 + lam2), lambda = c(1, lam1 + lam2),
                 intercept = FALSE, max.iter = 1000, eps = 1e-5)
bhat1 = as.numeric(res1$beta[, 2])
objfn(bhat1, x, y, kappa, lam1, lam2)

# ReHLine
bhat2 = elasticqr(x, y, kappa = kappa, lam1 = lam1, lam2 = lam2,
                  max_iter = 1000, tol = 1e-5)
objfn(bhat2, x, y, kappa, lam1, lam2)

# Compare estimates
plot(bhat1, bhat2)
abline(0, 1, col = "red")
