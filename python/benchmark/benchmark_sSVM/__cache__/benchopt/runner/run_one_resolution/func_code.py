# first line: 19
def run_one_resolution(objective, solver, meta, stop_val):
    """Run one resolution of the solver.

    Parameters
    ----------
    objective : instance of BaseObjective
        The objective to minimize.
    solver : instance of BaseSolver
        The solver to use.
    meta : dict
        Metadata passed to store in Cost results.
        Contains objective, data, dimension, id_rep.
    stop_val : int | float
        Corresponds to stopping criterion, such as
        tol or max_iter for the solver. It depends
        on the stopping_strategy for the solver.

    Returns
    -------
    cost : dict
        Details on the run and the objective value obtained.
    """
    # check if the module caught a failed import
    if not solver.is_installed():
        raise ImportError(
            f"Failure during import in {solver.__module__}."
        )

    t_start = time.perf_counter()
    solver.run(stop_val)
    delta_t = time.perf_counter() - t_start
    beta_hat_i = solver.get_result()
    objective_dict = objective(beta_hat_i)

    # Add system info in results
    info = get_sys_info()

    return dict(**meta, stop_val=stop_val, time=delta_t,
                **objective_dict, **info)
