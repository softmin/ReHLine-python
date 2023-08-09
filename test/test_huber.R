library(rehline)
library(hqreg)

# Simulate data
set.seed(123)
n = 5000
d = 100
x = matrix(rnorm(n * d), n, d)
beta0 = rnorm(d)
y = c(x %*% beta0) + rnorm(n, sd = 0.1)
kappa = IQR(y) / 10

lam1 = 0.05
lam2 = 0.05

huber = function(x, kappa)
{
    xabs = abs(x)
    ifelse(xabs <= kappa, xabs^2 / 2, kappa * (xabs - kappa / 2))
}

# Objective function value
objfn = function(beta, x, y, kappa, lam1, lam2)
{
    d = length(beta)
    pred = c(x %*% beta)
    loss = huber(y - pred, kappa)
    reg = lam1 * sum(abs(beta)) + lam2 * sum(beta^2) / 2
    mean(loss) + reg
}

# Reparameterization of penalty parameters
# lam1 = lambda * alpha
# lam2 = lambda * (1 - alpha)
# => alpha = lam1 / (lam1 + lam2), lambda = lam1 + lam2
res1 = hqreg_raw(x, y, method = "huber", gamma = kappa,
                 alpha = lam1 / (lam1 + lam2), lambda = c(1, lam1 + lam2),
                 intercept = FALSE, max.iter = 1000, eps = 1e-5)
bhat1 = as.numeric(res1$beta[, 2])
objfn(bhat1, x, y, kappa, lam1, lam2)

# ReHLine
bhat2 = elastic_huber(x, y, kappa = kappa, lam1 = lam1, lam2 = lam2,
                      max_iter = 1000, tol = 1e-5)
objfn(bhat2, x, y, kappa, lam1, lam2)

# Compare estimates
plot(bhat1, bhat2)
abline(0, 1, col = "red")
