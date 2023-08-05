# first line: 60
def run_one_to_cvg(benchmark, objective, solver, meta, stopping_criterion,
                   force=False, output=None, pdb=False):
    """Run all repetitions of the solver for a value of stopping criterion.

    Parameters
    ----------
    benchmark : benchopt.Benchmark object
        Object to represent the benchmark.
    objective : instance of BaseObjective
        The objective to minimize.
    solver : instance of BaseSolver
        The solver to use.
    meta : dict
        Metadata passed to store in Cost results.
        Contains objective, data, dimension.
    stopping_criterion : StoppingCriterion
        Object to check if we need to stop a solver.
    force : bool
        If force is set to True, ignore the cache and run the computations
        for the solver anyway. Else, use the cache if available.
    pdb : bool
        It pdb is set to True, open a debugger on error.

    Returns
    -------
    curve : list of Cost
        The cost obtained for all repetitions.
    status : 'done' | 'diverged' | 'timeout' | 'max_runs'
        The status on which the solver was stopped.
    """

    curve = []
    with exception_handler(output, pdb=pdb) as ctx:

        if solver._solver_strategy == "callback":
            output.progress('empty run for compilation')
            run_once_cb = _Callback(
                lambda x: {'objective_value': 1},
                {},
                stopping_criterion.get_runner_instance(
                    solver=solver, max_runs=1
                )
            )
            solver.run(run_once_cb)

            # If stopping strategy is 'callback', only call once to get the
            # results up to convergence.
            callback = _Callback(
                objective, meta, stopping_criterion
            )
            solver.run(callback)
            curve, ctx.status = callback.get_results()
        else:

            # Create a Memory object to cache the computations in the benchmark
            # folder and handle cases where we force the run.
            run_one_resolution_cached = benchmark.cache(
                run_one_resolution, force
            )

            # compute initial value
            call_args = dict(objective=objective, solver=solver, meta=meta)

            stop = False
            stop_val = stopping_criterion.init_stop_val()
            while not stop:

                cost = run_one_resolution_cached(stop_val=stop_val,
                                                 **call_args)
                curve.append(cost)

                # Check the stopping criterion and update rho if necessary.
                stop, ctx.status, stop_val = stopping_criterion.should_stop(
                    stop_val, curve
                )

    return curve, ctx.status
