# GDS r005 — Adversarial Verification Report

Task: task-261022-01. Package: `gds_src_pkg--r005--BMJ_20261030_002019.tar.gz`.

## Result

**PASS (software-only scope).** 77 Python unit/adversarial tests passed both in the development copy and after extracting the final release TAR in a clean directory. End-to-end no-GPU `suite selftest` passed in both locations. The source tree had **zero changed file hashes** after normal selftest+observe execution (before the intentionally version-stamping packaging step). The archive has **49 entries** and **30 BUILD_ID-stamped source/mermaid files**; no archive path traversal or symbolic links were found.

## Adversarial matrix

| Test | Expected defense | Verified outcome |
|---|---|---|
| `nvidia-fs` absent but Linux PCI P2PDMA possible | Do not falsely label compatibility | PASS (support/verdict tests) |
| Strict cuFile success without bus telemetry | Do not claim physical P2P proof | PASS (`NATIVE_GDS_CANDIDATE`) |
| Global NVFS counters moved | Avoid claiming per-process P2P proof | PASS (`NATIVE_GDS_INDICATED`) |
| Explicit cuFile compatibility report contradicts counters | Prefer explicit self-report, flag conflict | PASS |
| Forced fallback | Set `force_compat_mode=true` (not just allow fallback) | PASS (config fixture) |
| DGX Spark tries forced native | Reject known unsupported hardware configuration | PASS (mocked platform test) |
| Fake `gdscheck` line `NVMe P2PDMA: Unsupported` | Do not parse it as positive NVMe capability | PASS |
| Unmounted fake partition | Validate positive geometry/identity | PASS (sysfs/proc fixture, no real block writes) |
| Whole raw disk, mounted partition, swap, device holders | Fail closed | PASS (fixture tests) |
| Negative, unaligned, out-of-bound raw ranges | Fail closed | PASS |
| CLI `--raw-device /dev/null` with overwrite consent | Reject non-block target | PASS, exit 1 |
| CLI `--run-dir /tmp/outside-exe-out` | Prevent artifacts outside exe_out | PASS, exit 1 |
| CLI `mermaid --run-dir /tmp/outside-exe-out` | Reject external output path | PASS, exit 1 |
| CLI invalid KV sizes (`--sizes-kib 1,3`) | Reject sub-4KiB sizes | PASS, exit 1 |
| Host execution of `suite selftest` | No source files modified | PASS (SHA-256 source inventory comparison) |
| Full release TAR extraction and selftest | Works without development environment side effects | PASS |

## Threat model and limitations

1. Fixture checks prove the **Python safety rules** in isolation, not the behavior of a live mounted NVMe device during a race. A dedicated disposable partition and exclusive operator control are still required.
2. The CUDA `.cu` binary could not be compiled or executed because this verification environment has no `nvcc`/GPU/cuFile runtime. This report is **not** hardware certification for DGX Spark or RTX PRO 6000.
3. Physical PCIe DMA TLPs, IOMMU translations, GPU Fabric/Memory Controller operation and SSD DMA internals are not observable from BXTRACE/NVTX alone. `--trace-cmd` and `--nsys` were included but not hardware-validated here.
4. `gdscheck` support modes identify available **capabilities**, not evidence that a specific payload used direct P2P.
5. `root/.pyve` remains intentionally outside `root/exe_out` per the user's explicit exception; it is not an error.

## Command examples

```bash
cd root/src_pkg
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
./bx_gds_run_top.py suite selftest
./bx_gds_run_top.py observe
./bx_gds_run_top.py all --dry-run
```

Official references: NVIDIA [GDS Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html), [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/), [GDS Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/), [PCI P2PDMA Troubleshooting](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/), and Linux [PCI P2PDMA](https://docs.kernel.org/driver-api/pci/p2pdma.html).
