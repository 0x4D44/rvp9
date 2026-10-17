# Current-source rvp9-io Miri attempt

The full rvp9-io Miri test command exited 101 while compiling its rvp9-core dependency. Eight E0425 diagnostics report missing scalar transform helpers; the test harness did not start, so this attempt provides no parser or container test passes. The failure matches the dependent-library Miri configuration issue tracked by RVP9-BUG-FLU-00037.

The runner checkout is 44140dbb0e0834af296fac124d3a9293e486b53b, with production source fb5c0c917038db6c640b3dbc7c2b84b567233333. The tested rvp9-core, rvp9-io, and workspace manifest paths match that product source. Toolchain, command, error list, raw-log hash, host identity, and exit status are in qualification.json; compiler output is preserved in run.raw.txt.
