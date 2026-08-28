# Security

## What this project is

A desktop cloth simulation. It opens a window, talks to the GPU, and exits. It
has no network client or server, reads no files at runtime, takes no command
line arguments, stores nothing, and handles no credentials or personal data.
Every shader it compiles is bundled in the binary at build time.

So the usual deploy gate, authorization, input validation at a trust boundary,
secret handling, has little to bite on here. Being honest about that is more
useful than a checklist of controls that do not apply.

## What is actually worth reporting

- A crash, hang or unbounded allocation reachable from the interface. The
  simulation is bounded on purpose: the catch-up loop caps the steps a single
  frame may run, and the grid side and spacing are bounded by the panel. A way
  around either is a real defect.
- Anything in the dependency tree with a known advisory. `cargo audit` runs in
  CI on every push through `rustsec/audit-check`, and Dependabot opens pull
  requests for updates, so this should surface on its own. If it does not,
  please say so.
- Undefined behaviour. The crate has no `unsafe` block of its own; `bytemuck`
  derives the casts, which is why the CPU and GPU struct layouts are pinned by
  tests rather than trusted.

## Reporting

Open a [private security advisory](https://github.com/DeharengOlivier/gpu-cloth-simulation/security/advisories/new)
rather than a public issue, and allow a little time for a fix before disclosing.

## Supply chain

`wgpu-bootstrap` is a git dependency. It is requested by tag, and a tag can be
moved, so `Cargo.lock` is committed and pins the exact commit
(`a1df470`). Builds resolve to that commit until the lock file is deliberately
updated.
