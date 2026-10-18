# Wrong Y4M header preflight evidence

This run exercised the real scripts/qualify-production-encoder.sh control flow from task HEAD f55242e38a8d5ba8644ef60aa943469730e8643c. It is branch evidence only, not a production-input qualification: the scratch copy changes one expected corpus digest so the SHA-256 check accepts a deliberately wrong-header fixture and the real header rejection branch executes.

The script returned exit code 1 with the exact diagnostic preserved in raw-output.txt:

    wrong corpus Y4M header for in_to_tree_420_720p50.y4m: 'YUV4MPEG2 W1280 H719 F50:1 Ip A1:1' (expected 'YUV4MPEG2 W1280 H720 F50:1 Ip A1:1')

production-preflight-order.txt records the source lines showing SHA-256 before header comparison and mkdir -p "$work_dir" after all corpus checks. The external work directory was absent before and after. The four help-probed tools made 13 --help stub calls; the non-help stub log is empty. No quality or codec phase was reached.

corpus-verification.tsv records hashes and first lines for each input passed to the script. Three are canonical symlinked corpus inputs. The wrong-header fixture is a one-line Y4M header whose hash was installed in the scratch script copy. The canonical original in_to_tree source hash and header are also recorded there.

source-identity.txt, qualification-command.txt, scratch-script-mutation.patch, and tool-hashes.sha256 retain the source, exact invocation, one-line mutation, and tool identities. The help-only stub and wrong-header fixture are included so the branch setup is inspectable. The two invocation logs were scratch-only and were not retained in this bundle: the recorded help-log SHA-256 is `02e06f5288ac2a373cc0374215ceaf4fb2e3259f2a339413cb4f81caa441d501`, and the non-help-log SHA-256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (the empty-file digest). Their counts and zero non-help result remain in result-assertions.txt, but the raw logs cannot be independently rechecked here. artifact-hashes.sha256 hashes every retained file in this evidence bundle except itself using bundle-relative paths.

The modified script, fixture, and stubs used for execution lived only under this worktree's .scratch; the original production script and source corpus remain unchanged.
