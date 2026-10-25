# Source-built Spark stack in Docker (r20)

The public Docker entry point is `../bx_ai_coding_docker.sh`. The existing
`bx_ai_coding_setup.sh` remains the source-build controller. Their roles differ:
the Docker wrapper manages images/containers and invokes the same retained
controller inside the container. Source-stage numbering is unchanged.

This deployment reuses the **completed host source build**, including editable
Python packages, native build outputs, model weights and OpenCode V1. It is a
**path-preserving, bind-mounted runtime**, not a standalone portable image.
The runtime image supplies Ubuntu 24.04, CUDA 13, cuDNN, Python 3.12 and OS
libraries. It does not install binary replacements for the core projects.
Sources, logs, agent state and caches remain on the host after container exit.

## Prerequisites

- Native aarch64 DGX Spark / GB10, working NVIDIA host driver.
- Local Docker Engine and NVIDIA Container Toolkit configured for GPU access.
  The wrapper checks GPU access; it does not install Docker or restart its daemon.
- Completed host stages through `130_verify_build`, retained source and weights.
- Permission to access the local Docker Unix socket. Do not invoke the wrapper
  with sudo merely to bypass socket permissions: host UID mapping would change.
- Free unified memory for one server. Finish OpenCode requests and stop the host
  stack/baseline in its owning terminal before starting a container server.
- An existing writable project directory for serving/agent steps.

## Normal sequence

Extract the r20 package into a staging directory, preserve your active config,
and install the updated controller at `/opt/work/ai_coding/ai_coding_source`.
Do not overwrite a custom `configs/stack.json` with the bundled default. Do not
rebuild successful core projects for this runtime migration.

```bash
cd /opt/work/ai_coding/ai_coding_source
./bx_ai_coding_setup.sh 010a_validate_scripts
./bx_ai_coding_docker.sh --help
./bx_ai_coding_docker.sh 010_check_host --execute
./bx_ai_coding_docker.sh 020_build_image --execute
./bx_ai_coding_docker.sh 030_verify_runtime --execute
mkdir -p /opt/work/ai_coding/agent_sandbox
```

Terminal A:

```bash
./bx_ai_coding_docker.sh 040_serve_stack --project /opt/work/ai_coding/agent_sandbox --execute
```

Wait for `API READY`. Startup performs `130_verify_build` **inside the
container first**, including source-origin/native CUDA/model imports, pip
check and the actual interactive agent CLI contract. It fails on missing OS
libraries or broken editable paths rather than installing replacement wheels.

Terminal B:

```bash
cd /opt/work/ai_coding/ai_coding_source
./bx_ai_coding_docker.sh 050_test_api --execute
./bx_ai_coding_docker.sh 060_run_agent --project /opt/work/ai_coding/agent_sandbox --execute
```

Stop from Terminal A with Ctrl-C, or in another terminal:

```bash
./bx_ai_coding_docker.sh 070_stop --execute
```

`--rm` removes the stopped container, not the bind-mounted host files. It
also removes files written only into the container's writable layer. Container
HOME is temporary; OpenCode's configured state is under the persistent task
output tree. Unexpected Docker daemon/machine shutdown is not a graceful stop.

All commands except help default to printing a plan without writes, Docker
calls, downloads, or container execution. Add `--execute` to act. The container
is foreground, not a systemd service and not automatically restarted.

## Paths and permissions

The wrapper reads the same JSON profile as the build controller. It mounts:

- task root and its `realpath`, read/write;
- controller and profile directory, read-only;
- configured source/model directories and their real paths, read/write;
- selected project and its real path, read/write;
- explicit extra dependency directories, read-only.

Both `/opt/work/...` aliases and long `/home/...` canonical paths are retained.
This matters because editable `.pth` loaders, native build outputs, venv scripts
and Git worktrees can contain absolute host paths. The wrapper does not mount
the entire home directory or root filesystem. Avoid commas/control characters
in mount paths; spaces are supported. Use matching host UID/GID and supplementary
groups so generated project files retain host ownership. Shared memory is a
private Docker tmpfs with capacity 8g by default, adjustable via `--shm-size`.
This is not an 8 GiB allocation at startup, and does not change the 4 GB LMCache
L1 configuration.

A Python venv is not universally portable. The default image uses the same
Ubuntu/Python/CUDA family as the host, but additional native dependencies or
absolute symlink targets may be absent. The pre-start verification is mandatory.
If it fails, inspect its first error and image/host library versions before
changing anything. For a missing OS package, extend `docker/Dockerfile`, update
its BUILD_ID and rebuild the runtime image; do not replace source-built torch.
For differing cuDNN ABI, select a matching base image/library installation.

For an isolated source profile, use its same config in every terminal:

```bash
./bx_ai_coding_docker.sh 030_verify_runtime --config /opt/work/ai_coding/root/profiles/stable_20261024/stack.json --execute
```

Worktree `.git` files can point outside the mounted source/root. Inspect the
reported common Git directory on the host, then mount that exact dependency:

```bash
git -C /opt/work/ai_coding/root/src/pytorch rev-parse --path-format=absolute --git-common-dir
./bx_ai_coding_docker.sh 030_verify_runtime --extra-mount /absolute/path/to/common/git-directory --execute
```

The example path must be replaced with the actual result. Use the same extra
mounts at server launch. If editable links depend on another directory, review
and mount that directory explicitly. Projects outside the selected project/root
are not automatically made available to agent tools. Stop/restart the server
with the new `--project` to change projects; exec refuses an unmounted project.
Do not run host and container source builds simultaneously against the same
writable source/venv. This wrapper is for runtime, not parallel rebuilds.

## Networking and lifecycle

`--network host` preserves localhost:8000/5555/8080 and the generated provider
configuration. There are no published-port mappings. These processes still
share host port space: containers do not resolve an existing port collision.
API bindings remain 127.0.0.1 through the retained controller. This design is for
a local development machine; host networking is not network isolation.

The wrapper uses `--gpus all`, `--init`, `--stop-signal SIGINT`, stop timeout
60 seconds and the current host UID/GID. The serving controller is exec'd as
the init process's direct child, so its KeyboardInterrupt cleanup can stop its
owned vLLM/LMCache process groups. There is no privileged mode, Docker socket
mount, host PID namespace, or automatic broad process kill. Stop/exec checks
ownership plus config identity labels and refuses foreign containers. Docker
itself refuses a duplicate container name. Multiple image/config combinations
must use different names, but the default inference ports still collide.

## Diagnosis, baseline and API profiling

```bash
./bx_ai_coding_docker.sh 050b_logs --execute
./bx_ai_coding_docker.sh 060a_shell --execute
```

Detailed vLLM/LMCache logs remain in host
`ROOT/deliverables/ai_coding/{vllm,lmcache}.log`; docker logs shows controller
stdout/stderr and startup failure tails, not every live service log line.

For baseline, stop the stack and launch the alternative in Terminal A:

```bash
./bx_ai_coding_docker.sh 040a_serve_baseline --project /opt/work/ai_coding/agent_sandbox --execute
```

After readiness, benchmark from Terminal B with an appropriate mode label and
new output directory:

```bash
./bx_ai_coding_docker.sh 050a_profile --execute -- --action benchmark --mode baseline --output /opt/work/ai_coding/root/deliverables/ai_coding/perf/docker_baseline
```

Switch servers, then measure with `--mode lmcache` and a new output. Analyze
and compare via the same wrapper while the managed container is running.
`--mode` remains a declaration, not proof of the live configuration. Baseline
and stack preserve the host's BF16, eager, max length and LMCache settings.
CPU L1 caching is unchanged; SSD/GDS is still NOT_CONFIGURED.

Only benchmark/analyze/compare are forwarded. Nsight **server launch** inside
this wrapper is intentionally not implemented. Profilers need a separately
reviewed image/tool installation, driver counter permissions and process scope.
A nonprivileged container may lack Nsight tools or hardware-counter access;
this package does not silently grant broad capabilities. For initial native
Nsight capture, stop the Docker server and use the existing host profiling
workflow in `docs/performance/Profiling_Guide.md`.

Container pidstat sees container processes, whereas some procfs/CPU/memory
metrics may reflect host-wide state. Collect host vmstat/mpstat/iostat as well
when diagnosing end-to-end contention. Docker is not a throughput optimization
claim; repeat the same direct API workload to measure overhead.

## Image selection and provenance

Default base: `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04`. Docker builds native
`linux/arm64`. Check registry availability on the target. The default is a tag,
not a content digest. For repeatable runtime OS/toolkit contents, record the
resolved image ID and select an available pinned digest:

```bash
./bx_ai_coding_docker.sh 020_build_image --base-image YOUR_REVIEWED_IMAGE_DIGEST --image bx-ai-coding:reviewed --execute
docker image inspect bx-ai-coding:reviewed
```

Use the same `--image` for verification and server launch. Do not assume saving
this runtime image also saves host sources, venv or model weights: archive those
separately with locks, source snapshots and LFS data. Full relocation to another
machine requires rebuilding at reviewed paths or a container-native source-build
migration; this package does not claim either has already been validated.

## Verification status

CPU/offline command and guard tests are included in `tests/test_docker.py` and
run by the usual 010a validation. This development environment has no Docker
Engine or GB10; image pull/build, real CUDA imports and live Docker inference
are **NOT_RUN** here. A successful 030 on Spark proves runtime imports/smoke;
050 proves structured API tool-call output; actual OpenCode tasks, external
cache hits and production performance remain separate checks.

References:
- https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html
- https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/compilation.html
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- https://docs.docker.com/engine/storage/bind-mounts/
- https://docs.docker.com/engine/network/drivers/host/
- https://docs.docker.com/engine/containers/run/
