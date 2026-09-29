"""Regression checks for the coordinate-descent solver's dual descent."""

import re

import numpy as np
import pytest

from rehline import ReHLine_solver


@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("coordinate_order", ["auto", "cyclic", "random"])
@pytest.mark.parametrize("constrained", [False, True])
def test_ill_conditioned_dual_does_not_worsen_after_more_sweeps(shrink, coordinate_order, constrained):
    # B = -X.T has nearly opposite columns. Forming B.T @ B in float64
    # loses the unit curvature in its first diagonal entry (1e16 + 1).
    # The removed block update consequently accepted a dual objective rise
    # of 0.1249 at sweep 50, or sweep n when constraints were present.
    n = 75 if constrained else 2
    X = np.zeros((n, 2))
    X[:2] = [[-1e8, -1.0], [1e8, 0.0]]
    U, V, initial = (np.zeros((1, n)) for _ in range(3))
    U[0, :2] = 1.0
    V[0, :2] = [0.5001, 0.0001]
    initial[0, :2] = 0.5
    options = dict(A=np.array([[0.0, 1.0]]), b=np.array([1.0]), xi=np.zeros(1)) if constrained else {}
    period = n if constrained else 50

    # For lambda1 = lambda2 = t, the minimized negative dual is
    # F(t) = t**2 / 2 - 0.5002*t. The warm start has F(0.5) = -0.1251.
    previous = -0.1251
    for sweeps in (period - 1, period, 2 * period):
        result = ReHLine_solver(
            X,
            U,
            V,
            Lambda=initial,
            max_iter=sweeps,
            tol=1e-8,
            shrink=shrink,
            coordinate_order=coordinate_order,
            verbose=0,
            **options,
        )
        beta = -X.T @ (U * result.Lambda).sum(axis=0)
        if constrained:
            beta += options["A"].T @ result.xi
        negative_dual = 0.5 * beta @ beta - np.sum(V * result.Lambda)
        if constrained:
            negative_dual += options["b"] @ result.xi
        assert negative_dual <= previous + 1e-12
        assert result.dual_objective == pytest.approx(-negative_dual, abs=1e-12)
        # Precision-limited stagnation must remain visible to the caller.
        assert not result.converged
        previous = negative_dual


def shrinking_problem(*, degenerate_groups=False, weight=3.0):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(48, 3))
    y = np.where(X[:, 0] + 0.5 * rng.normal(size=48) > 0, 1.0, -1.0)
    p = dict(X=X, U=-weight * y[None, :], V=np.full((1, 48), weight))
    if degenerate_groups:
        # Zero-curvature constraints have no PG samples; zero ReHU bounds
        # can leave an empty working set. Neither may disable early recovery.
        p.update(A=np.zeros((1, 3)), b=np.ones(1), S=np.zeros((1, 48)), T=np.zeros((1, 48)), Tau=np.zeros((1, 48)))
    return p


@pytest.mark.parametrize("order", ["cyclic", "random"])
@pytest.mark.parametrize("degenerate_groups", [False, True])
def test_relaxed_shrinking_restores_before_final_accuracy(order, degenerate_groups, capfd):
    tol = 1e-8
    p = shrinking_problem(degenerate_groups=degenerate_groups)
    # A local recovery threshold must never be used to certify this unfinished
    # fit. Capture native diagnostics to check that early recovery actually ran.
    options = dict(shrink=1, coordinate_order=order, coordinate_seed=7, tol=tol, verbose=1, trace_freq=1)
    ReHLine_solver(**p, **options, max_iter=100)
    messages = capfd.readouterr().out
    recoveries = re.findall(
        r"Iter (\d+), restoring all variables; local_pg = ([^,\n]+), eps_shrink = ([^\n]+)", messages
    )
    early = [int(i) for i, pg, eps in recoveries if tol < float(pg) < float(eps)]
    assert early, messages
    # Replay the prefix ending at the first relaxed recovery request.
    result = ReHLine_solver(**p, **options, max_iter=early[0] + 1)
    assert not result.converged
    assert result.kkt_residual > tol
    primal = 0.5 * result.beta @ result.beta + np.maximum(p["U"] * (p["X"] @ result.beta) + p["V"], 0).sum()
    dual_beta = -p["X"].T @ (p["U"] * result.Lambda).sum(axis=0)
    dual = (p["V"] * result.Lambda).sum() - 0.5 * dual_beta @ dual_beta
    assert (primal - dual) / max(1, abs(primal), abs(dual)) > tol
    trace = np.asarray(result.dual_objfns)
    assert np.all(np.diff(trace) <= 1e-12 * np.maximum(1, abs(trace[:-1])))


@pytest.mark.parametrize("order", ["cyclic", "random"])
@pytest.mark.parametrize("weight", [0.03, 3.0, 30.0])
def test_shrinking_keeps_full_certificate_across_loss_scales_and_warm_fits(order, weight):
    p = shrinking_problem(degenerate_groups=True, weight=weight)
    options = dict(shrink=1, coordinate_order=order, coordinate_seed=7, tol=1e-8, max_iter=1_000_000, verbose=0)
    result = ReHLine_solver(**p, **options)
    for _ in range(2):
        assert result.converged
        assert np.isfinite(result.kkt_residual)
        assert result.scaled_constraint_violation == 0
        primal = 0.5 * result.beta @ result.beta + np.maximum(p["U"] * (p["X"] @ result.beta) + p["V"], 0).sum()
        dual_beta = -p["X"].T @ (p["U"] * result.Lambda).sum(axis=0)
        dual = (p["V"] * result.Lambda).sum() - 0.5 * dual_beta @ dual_beta
        assert -1e-12 <= (primal - dual) / max(1, abs(primal), abs(dual)) <= options["tol"]
        np.testing.assert_allclose(result.beta, dual_beta, atol=1e-8, rtol=0)
        assert np.all((result.Lambda >= 0) & (result.Lambda <= 1))
        assert np.all(result.Gamma == 0) and np.all(result.xi == 0)
        assert result.objective == pytest.approx(primal, rel=1e-12, abs=1e-12)
        assert result.dual_objective == pytest.approx(dual, rel=1e-12, abs=1e-12)
        warm = {field: getattr(result, field) for field in ("Lambda", "Gamma", "xi", "mu")}
        result = ReHLine_solver(**p, **warm, **options)


@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("order", ["cyclic", "random"])
@pytest.mark.parametrize("quantiles", [0, 2])
@pytest.mark.parametrize("constrained", [False, True])
def test_objective_certificate_stops_without_absolute_kkt_accuracy(shrink, order, quantiles, constrained):
    # Two nearly redundant ReHU dual coordinates can exchange mass for a long
    # time although the primal objective is already accurate. Their raw PG is
    # about 1e-6, while the independently evaluated bound gap is below 2e-12.
    a, t, tol = 1e4, 100.0, 1e-8
    X = np.full((1 if quantiles else 2, 1), a)
    design = np.column_stack((np.full(2, a), np.eye(2))) if quantiles else X
    options = dict(A=np.eye(1, design.shape[1]), b=np.ones(1), xi=np.zeros(1)) if constrained else {}
    result = ReHLine_solver(
        X,
        None,
        None,
        S=np.ones((1, 2)),
        T=np.full((1, 2), t),
        Tau=np.full((1, 2), np.inf),
        Gamma=np.array([[1e-6, 0.0]]),
        tol=tol,
        max_iter=100,
        verbose=0,
        shrink=shrink,
        coordinate_order=order,
        coordinate_seed=7,
        _quantile_count=quantiles,
        **options,
    )
    dual_beta = -design.T @ result.Gamma[0]
    if constrained:
        dual_beta += options["A"].T @ result.xi
        assert np.all(options["A"] @ result.beta + options["b"] >= 0)
    loss = np.maximum(design @ result.beta + t, 0)
    primal = 0.5 * result.beta @ result.beta + 0.5 * loss @ loss
    dual = t * result.Gamma.sum() - 0.5 * np.square(result.Gamma).sum() - 0.5 * dual_beta @ dual_beta
    if constrained:
        dual -= options["b"] @ result.xi
    optimum = t**2 / (1 + 2 * a**2 + bool(quantiles))
    assert dual <= optimum <= primal
    assert 0 <= (primal - dual) / max(1, abs(primal), abs(dual)) <= tol
    assert result.objective == pytest.approx(primal, abs=1e-15, rel=1e-12)
    assert result.dual_objective == pytest.approx(dual, abs=1e-15, rel=1e-12)
    assert result.kkt_residual > 50 * tol
    assert result.converged
    assert result.niter == 1  # The loop itself terminates; no max_iter relabeling.


@pytest.mark.parametrize("shrink", [0, 1])
def test_budget_end_uses_same_objective_certificate(shrink):
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    initial = np.array([[0.4, 0.20001]])
    r = ReHLine_solver(
        X,
        None,
        None,
        S=np.ones((1, 2)),
        T=np.ones((1, 2)),
        Tau=np.full((1, 2), np.inf),
        Gamma=initial,
        tol=1e-8,
        max_iter=1,
        shrink=shrink,
        coordinate_order="cyclic",
        verbose=0,
    )
    assert np.linalg.norm(r.beta + X.T @ initial[0]) > 1e-8
    assert r.kkt_residual > 1e-8  # Neither cheap loop trigger passes at this budget.
    primal = 0.5 * r.beta @ r.beta + 0.5 * np.square(np.maximum(X @ r.beta + 1, 0)).sum()
    dual_beta = -X.T @ r.Gamma[0]
    dual = r.Gamma.sum() - 0.5 * np.square(r.Gamma).sum() - 0.5 * dual_beta @ dual_beta
    assert dual <= 0.3 <= primal
    assert 0 <= primal - dual <= 1e-8
    assert r.converged and r.niter == 1


@pytest.mark.parametrize("order", ["cyclic", "random"])
@pytest.mark.parametrize("constrained", [False, True])
def test_recovery_event_stops_on_global_gap_and_feasibility(order, constrained, capfd):
    p = shrinking_problem()
    if constrained:
        p.update(A=np.array([[1.0, 0.0, 0.0]]), b=np.array([10.0]))
    tol = 1e-8
    r = ReHLine_solver(
        **p, tol=tol, max_iter=1000, shrink=1, coordinate_order=order, coordinate_seed=7, verbose=2, trace_freq=1
    )
    messages = capfd.readouterr().out
    stopped = re.findall(r"Iter (\d+), global gap and feasibility passed before restoration", messages)
    assert len(stopped) == 1 and int(stopped[0]) + 1 == r.niter
    active, total = re.findall(r"lambda \((\d+)/(\d+)\)", messages)[-1]
    assert int(active) < int(total)  # This certificate covers shrunken coordinates too.
    beta = -p["X"].T @ (p["U"] * r.Lambda).sum(axis=0)
    linear = 0.0
    if constrained:
        beta += p["A"].T @ r.xi
        linear = p["b"] @ r.xi
        assert np.all(p["A"] @ r.beta + p["b"] >= 0)
    primal = 0.5 * r.beta @ r.beta + np.maximum(p["U"] * (p["X"] @ r.beta) + p["V"], 0).sum()
    dual = (p["V"] * r.Lambda).sum() - 0.5 * beta @ beta - linear
    assert np.all((r.Lambda >= 0) & (r.Lambda <= 1))
    np.testing.assert_allclose(r.beta, beta, atol=1e-12, rtol=1e-12)
    assert -1e-12 <= (primal - dual) / max(1, abs(primal), abs(dual)) <= tol
    assert r.objective == pytest.approx(primal, abs=1e-12)
    assert r.dual_objective == pytest.approx(dual, abs=1e-12)
    assert r.kkt_residual > tol
    assert r.converged and r.niter < 1000


@pytest.mark.parametrize("order", ["cyclic", "random"])
@pytest.mark.parametrize("scales", [[1.0, 1.0], [1e-100, 1e100]])
def test_recovery_rejects_small_gap_when_constraints_are_violated(order, scales, capfd):
    p = shrinking_problem()
    # Impossible pair: beta[0] >= .001 and beta[0] <= 0. A large constant
    # loss makes the relative objective difference small without feasibility.
    p["U"] = np.vstack((p["U"], np.zeros((1, 48))))
    p["V"] = np.vstack((p["V"], np.full((1, 48), 1e12)))
    scales = np.asarray(scales)
    A, b = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), np.array([-0.001, 0.0])
    p.update(A=A * scales[:, None], b=b * scales)
    r = ReHLine_solver(
        **p, tol=1e-8, max_iter=100, shrink=1, coordinate_order=order, coordinate_seed=7, verbose=1, trace_freq=1000
    )
    messages = capfd.readouterr().out
    assert "restoring all variables" in messages
    assert "passed before restoration" not in messages
    violation = max(0, -(A @ r.beta + b).min())
    assert violation >= 0.0005
    assert r.scaled_constraint_violation == pytest.approx(violation, abs=1e-12)
    assert abs(r.objective - r.dual_objective) / max(1, abs(r.objective), abs(r.dual_objective)) < 1e-8
    assert not r.converged and r.niter == 100
    assert np.isinf(r.dual_gap)


@pytest.mark.parametrize("order", ["cyclic", "random"])
@pytest.mark.parametrize("degenerate_groups", [False, True])
def test_recovery_gap_budget_counts_visited_coordinates(order, degenerate_groups, capfd):
    p = shrinking_problem(degenerate_groups=degenerate_groups)
    # Keep every certificate false in this short prefix. Reconstruct work from
    # independently printed active sets, including coordinates just removed.
    r = ReHLine_solver(
        **p, tol=1e-30, max_iter=40, shrink=1, coordinate_order=order, coordinate_seed=7, verbose=2, trace_freq=1
    )
    messages = capfd.readouterr().out
    budgets = [int(x) for x in re.findall(r"recovery gap probe; coordinate_budget = (\d+)", messages)]
    assert budgets and len(set(budgets)) == 1
    budget = budgets[0]
    full = p["U"].size + p.get("S", np.empty(0)).size + len(p.get("b", []))
    assert budget > full and budget % full == 0
    work, incoming, deferred, probes = 0, full, 0, 0
    iterations = re.split(r"(?m)^Iter \d+, ", messages)[1:]
    assert len(iterations) == r.niter == 40
    for iteration in iterations:
        work = min(budget, work + incoming)
        counts = re.search(r"xi \((\d+)/\d+\), lambda \((\d+)/\d+\), gamma \((\d+)/\d+\)", iteration)
        outgoing = sum(map(int, counts.groups()))
        if "recovery gap probe;" in iteration:
            assert work == budget
            work = 0
            probes += 1
        if "recovery gap probe deferred;" in iteration:
            actual, limit = map(int, re.search(r"coordinate_work = (\d+)/(\d+)", iteration).groups())
            assert actual == work < limit == budget
            # Deferring the certificate must not defer the full-set recovery.
            assert "restoring all variables" in iteration
            deferred += 1
        incoming = full if "restoring all variables" in iteration else outgoing
    assert probes > 0
    assert "passed before restoration" not in messages


def test_deferred_probe_still_restores_after_full_set_certificate(capfd):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 40))
    y = np.where(X[:, 0] + 0.5 * rng.normal(size=80) > 0, 1.0, -1.0)
    r = ReHLine_solver(
        X,
        -y[None, :],
        np.ones((1, 80)),
        tol=1e-8,
        max_iter=1000,
        coordinate_order="random",
        coordinate_seed=7,
        verbose=2,
        trace_freq=1,
    )
    messages = capfd.readouterr().out
    events = re.findall(r"Iter (\d+), recovery gap probe deferred; coordinate_work = (\d+)/(\d+)", messages)
    assert events
    for iteration, work, budget in events:
        assert int(work) < int(budget)
        assert f"Iter {iteration}, restoring all variables" in messages
        # The next sweep really visits the full set; it is not just a log.
        following = re.search(rf"(?m)^Iter {int(iteration) + 1}, .*\n([^\n]+)", messages)
        assert following and "lambda (80/80)" in following[1]
    primal = 0.5 * r.beta @ r.beta + np.maximum(1 - y * (X @ r.beta), 0).sum()
    beta = X.T @ (y * r.Lambda[0])
    dual = r.Lambda.sum() - 0.5 * beta @ beta
    assert r.converged and (primal - dual) / max(1, abs(primal), abs(dual)) <= 1e-8
