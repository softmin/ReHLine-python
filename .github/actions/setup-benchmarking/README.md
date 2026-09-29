# Private benchmark checkout

`softmin/ReHLine-benchmarking` is private. The default `GITHUB_TOKEN` is scoped
to `ReHLine-python` and cannot read that repository.

The action uses a dedicated SSH deploy key:

- Register the public key in `ReHLine-benchmarking` with **Allow write access**
  disabled.
- Store the private key in the `ReHLine-python` Actions repository secret
  `REHLINE_BENCHMARKING_SSH_KEY`.
- Pass the secret through the action's `ssh-key` input. Reusable workflow callers
  must also explicitly pass this secret to `ci.yml`.

Checkout uses `persist-credentials: false`, so later test and build steps do not
retain the SSH key. `REHLINE_BENCHMARKING_REF` remains an optional repository
variable selecting a branch, tag, or commit; it defaults to `main`.

To rotate the credential, register a new read-only deploy key, replace the
Actions secret, verify checkout, then revoke the old deploy key.

Fork pull requests do not receive repository secrets and cannot run checks that
require the private harness. Maintainers must review such contributions and run
these checks from a trusted repository branch. Keep the workflow on
`pull_request`; do not switch to `pull_request_target` to expose this credential
to untrusted code.
