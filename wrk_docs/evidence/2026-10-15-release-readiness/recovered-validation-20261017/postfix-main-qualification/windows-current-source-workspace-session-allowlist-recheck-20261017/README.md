# Current Windows workspace-session route recheck

The Windows full-workspace suite did not start. Four Deltic session-start attempts for the rvp9 repo key were rejected on ANNEAL, NMI, CRUCIBLE, and KILN before a session was created. No Windows host, toolchain, checkout, or test was preflighted. This is a routing limitation, not a test failure.

The remote Deltic interface does not forward host session-repository listings, so the exact host allowlists remain uninspectable through the approved route. The fleet roster query found no active RVP9 agent session to reuse. See qualification.json for the exact commands, responses, candidate identity, and observation time; the earlier launch record remains linked separately.
