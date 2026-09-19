# Benchmark results

Benchmark runners write generated reports here by default. New outputs are
ignored by Git and excluded from source distributions; CI retains its outputs
as workflow artifacts. Each report describes its stated source revision and
environment; it is not automatically a result for the current checkout.

Publish a report only after reviewing its source revision, configuration,
environment, objective and feasibility checks, convergence status, and treatment
of failed fits. A timing table without objective validation is not evidence that
solvers or releases reached equivalent solutions.

Exploratory comparisons, diagnostic retries and development notes belong in
the Git-ignored `local/` directory. They are not maintained benchmark results.
