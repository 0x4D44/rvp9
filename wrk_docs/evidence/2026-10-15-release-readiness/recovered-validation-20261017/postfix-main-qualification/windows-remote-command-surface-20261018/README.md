# Windows remote execution route check

At `2026-10-17T23:09:04Z`, Deltic help showed that `deltic remote` only forwards typed commands (`scratch`, `quota`, API status, and session start). It does not offer arbitrary remote command execution. The local `deltic session` help states that session repo allowlists are local to the target host and are not forwarded remotely. The fleet roster reported no active RVP9 session to reuse.

This is a control-surface check, not a Windows test result. It does not inspect NMI’s allowlist, change host configuration, or claim a code pass/failure. The Windows suite still needs an allowed repo session or another authorized execution route. See `qualification.json` for command results and raw hashes.
