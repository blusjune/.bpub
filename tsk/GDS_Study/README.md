# HANDOFF: GDS 실험 패키지 인수인계

이 문서는 지금까지의 작업을 새 세션에서 바로 이어 갈 수 있도록 정리한 것입니다.
지시 사항, 현재 상태, 실제로 배운 점, 미확인 항목, 다음 작업을 구분해서 적었습니다.

---

## 1. 목적과 범위

- 과제: task-261022-01 "Understanding GDS (GPU Direct Storage)"
- 산출물:
  1. GDS 기술 보고서 (markdown, mermaid 다이어그램 포함)
  2. GDS 동작 핵심을 보여 주는 CUDA 코드 (각 줄 주석, BXTRACE trace 로그)
  3. trace 로그에서 mermaid sequence diagram을 생성하는 Python 스크립트
  4. 아키텍처 계층도와 시나리오별 sequence diagram
  5. 실험 패키지 (실행 스크립트, 설정, 결과 디렉터리 규칙)
- 관심 주제: LLM inference에서 KV cache offload/onboard에 GDS를 어떻게 쓰는가
- 사용자 배경 (memory 기준): GPU 아키텍처, LLM inference 인프라, KV cache, storage 시스템 전문가. 1차 자료(arXiv, IEEE, SemiAnalysis)와 인용을 기대함.

---

## 2. 사용자가 명시적으로 지시한 내용

### 2.1 디렉터리와 실행 구조
- 모든 소스, 스크립트, 보고서, DoE, 테스트는 `./root/src_pkg/` 아래 적절한 하위 디렉터리에 둔다.
- 실행 결과는 `./root/exe_out/` 아래에 둔다. src_pkg와 exe_out은 완전히 격리한다.
- 실험 스크립트는 하나의 top-level script가 호출한다. 예: `bx_gds_run_top.py`가 subtest 스크립트를 호출.
- top-level script의 `--help`에 전형적인 사용 예시를 넣는다.

### 2.2 BXTRACE 계측
- 매크로 이름은 `BXTRACE()`.
- 현재 코드의 수행 위치를 trace log에서 구분할 수 있어야 한다. CPU, GPU, 기타 subsystem 구분.
- NVTX 또는 printf로 trace log 생성.
- 이 trace log로 sequence diagram mermaid script를 만드는 Python script 작성.
- 이후 추가 요청으로:
  - 각 event line에 소스 파일명과 줄 번호 포함 → sequence diagram의 각 event에 주석/note로 표시.
  - 모든 activity owner 이름에 실행 위치 표기 (CPU, GPU, SSD 등).
  - CUDA runtime 같은 호출 주체는 "코드가 실행되는 위치" 기준으로 표기. 메모리 객체는 "memory: 위치"로 표기.

### 2.3 CUDA 실험 코드
- GDS 동작 시나리오를 필요한 만큼 정의하고(IO init, GPU→SSD, SSD→GPU 등) 이를 구현하는 CUDA 코드 작성.
- 각 code line마다 상세 주석 (요청 원문: "각 code line마다 상세한 설명을 comment로서"). 현재 버전은 줄마다가 아니라 블록 단위 주석이므로 미충족 부분 있음 (아래 6장).
- GPU 내부 subsystem, CPU, MMU, IOMMU, PCIe root, SSD DMA engine의 역할을 trace로 알 수 있게 할 것. 단, printf/NVTX로는 GPU 내부 subsystem을 직접 알 수 없음 (4장 참조).

### 2.4 두 실험 노드
- 노드 A: DGX Spark (NVIDIA GB10, aarch64, Ubuntu 24.04.4, kernel 6.17.0-1014-nvidia, driver 580.126.09, CUDA 13.0). 사용자 말: GDS는 compatibility mode로만 동작함을 이미 알고 있음. 기본 실험부터 진행.
- 노드 B: Intel CPU + NVIDIA RTX PRO 6000 + Ubuntu 24.04. 실제 GDS 네이티브 동작을 보고 싶다고 함.
- 같은 소스를 두 노드에서 돌리되, 실행 경로가 native인지 compat인지 로그에서 구분 가능해야 함.

### 2.5 추가 요청 (시나리오)
- file system으로 data를 관리하는 상황의 GDS (init, read, write, ...)
- raw block mode로 data를 관리하는 상황의 GDS (init, read, write, ...)
- 각 시나리오가 독립적으로 trace log와 sequence diagram에서 구분되어야 함.

### 2.6 응답 형식 (memory preference, 기술 deep-dive용)
- 한국어 응답.
- 결론 먼저(conclusions-first).
- 약어는 처음 나올 때 풀어 쓴다.
- mermaid 다이어그램 포함.
- 1차 자료 인용 (URL 포함). 검색으로 확인한 내용만 인용하고, 확인하지 못한 내용은 미확인으로 표시.
- 여러 전문가 관점에서 교차 검토하는 수준의 엄밀함.

### 2.7 사용자가 확인을 요청했던 질문들
- GDS가 NVMe 외에 SATA HDD/SSD에서 되는가 → 3.2절.
- CUDA 코드의 CHECK_CUDA() 역할 → 설명 완료.
- 웜업이 무엇이며 왜 하는가 → 설명 완료.
- CUDA 코드 전체 동작 설명 → 설명 완료.

---

## 3. 현재 패키지 상태

파일 위치: `root/src_pkg/` (배포 파일명 `gds_src_pkg.tar.gz`)

| 파일 | 역할 | 상태 |
|---|---|---|
| `bx_gds_run_top.py` | top-level 드라이버 (env, build, roundtrip, all, mermaid) | 문법 검사 통과, Spark에서 실행 확인은 build 단계 실패 이후 미완 |
| `include/bx_trace.h` | BXTRACE, BXTRACE_LOC, NVTX 범위 매크로, 장치 record 구조체 | 호스트 쪽 컴파일 미확인 |
| `include/bx_trace_dev.cuh` | BX_DEV_PROBE: %smid, %warpid, %laneid, %globaltimer 읽기 | 컴파일 미확인 |
| `src/bx_gds_basic.cu` | 4개 시나리오(FS_GDS, FS_POSIX, RAW_GDS, RAW_POSIX) round trip, phase, probe | 컴파일 미확인 |
| `subtests/subtest_env_dump.py` | 호스트/GPU/PCIe/GDS 상태 덤프 | 실행 확인 (env 단계 결과는 사용자 측) |
| `subtests/subtest_roundtrip.py` | 시나리오 실행, cufile.json 생성, trace/summary 저장 | 문법 검사 통과 |
| `trace2mermaid.py` | trace.log → mermaid sequence diagram (시나리오별 파일, phase 블록, file:line 주석/note, 실행 위치 participant) | 단위 테스트 통과, Spark trace로 일부 실행 확인 |
| `tests/test_trace2mermaid.py` | 생성기 단위 테스트 | 18개 중 해당 항목 통과 |
| `tests/test_scenarios.py` | 시나리오 이름과 raw 안전장치 테스트 | 통과 |
| `docs/diagrams/architecture_layers.mmd` | 계층 구조 (미검증 노드 표시) | 작성 완료 |
| `docs/diagrams/scenario_gds_write.mmd`, `scenario_gds_read.mmd` | 일반 GDS write/read 가설 흐름 | 가설 |
| `docs/diagrams/scenario_posix_bounce.mmd` | POSIX bounce (코드와 일치) | 작성 완료 |
| `docs/diagrams/fs_gds_scenario.mmd` | FS_GDS 시나리오, phase 블록 | 가설 부분 포함 |
| `docs/diagrams/raw_gds_scenario.mmd` | RAW_GDS 시나리오, phase 블록 | cuFile raw 지원 미검증 |
| `docs/doe/DOE.md` | 두 노드 실험 설계 (인자, 반복, 응답) | 계획 |
| `docs/report/GDS_technical_report.md` | 보고서 초안, [CITE] 항목 있음 | 초안 |
| `README.md` | 파일 목록, trace 형식, 실행 순서 | 최신 |

### 3.1 trace 로그 형식 (현재)
```
Host:  BXTRACE|<monotonic_us>|CPU:core=<n>:tid=<t>|<component>|<step>|<file>:<line>|<message>
SM:    BXTRACE|<host_emit_us>|GPU:sm=<s>:warpslot=<w>:lane=<l>|SM|ProbeWarp|<file>:<line>|label=... tag=... gtimer_ns=...
Phase: BXTRACE|...|CPU_APP|Phase|<file>:<line>|name=<init|write|read|verify|cleanup>
Main:  BXTRACE|...|CPU_APP|Main|<file>:<line>|scenario=<S> storage=<fs|raw> mode=<gds|posix> target=... size=... iters=... offset=...
```
구 형식(파일:줄 없는 6필드)도 파서가 읽는다.

### 3.2 실행 명령 (현재 버전)
```bash
./bx_gds_run_top.py env
./bx_gds_run_top.py all --workdir <nvme mount> --size 1073741824 --iters 10 --drop-caches --nvtx
./bx_gds_run_top.py all --workdir <fs mount> --raw-device /dev/<node> --raw-offset 0 \
    --size 1073741824 --iters 10 --i-accept-overwrite
./bx_gds_run_top.py roundtrip --storage fs|raw --mode gds|posix ...
./bx_gds_run_top.py mermaid --run-dir ../exe_out/<host>/<ts>
python3 -m unittest discover -s tests -v
```

### 3.3 결과 디렉터리 (현재)
```
exe_out/<host>/<timestamp>/
  env/            env 덤프와 index.json
  build/          bx_gds_basic, build.log, build_command.txt
  FS_GDS/         trace.log, run.log, summary.json, cufile.json, trace_sequence_FS_GDS.mmd
  FS_POSIX/       ...
  RAW_GDS/        ... (raw 옵션 시)
  RAW_POSIX/      ...
  manifest.json
```

---

## 4. 실제로 배운 점 (경험과 교훈)

### 4.1 환경과 빌드
- `/bin/sh`는 중괄호 확장을 하지 않는다. `mkdir -p dir/{a,b}`는 `bash`에서만 동작한다. 실패 시 `{a,b}`라는 이름의 디렉터리가 생긴다. 쉘 스크립트에서는 명시적 경로를 쓴다.
- cuFile 구조체 필드 이름은 추측하면 안 된다. 실제 `cufile.h`를 `grep -A40 CUfileDrvProps`로 확인해야 한다. 추측으로 쓴 `props.nvfs.majorVersion`은 컴파일 오류가 났고, 실제 이름은 `nvfs.major_version`이다.
- `cuFileHandleDeregister()`는 `void`를 반환한다. `CUfileError_t`를 기대하는 래퍼에 넣으면 컴파일 오류.
- `/usr/local/cuda/gds/cufile.json`과 `/etc/cufile.json`에는 `//` 주석이 있어서 표준 JSON 파서가 실패한다. 주석 줄을 제거한 뒤 파싱하고, 설정을 덮어쓸 때는 기존 설정을 기준으로 해야 한다. 설정 파일을 키 두 개만으로 통째로 대체하면 나머지 기본값이 어떻게 채워지는지 보장되지 않는다.
- `cufile.json` 실제 키 (확인됨): `properties.allow_compat_mode`, `properties.use_pci_p2pdma`, `logging.dir`, `logging.level`, `profile.nvtx`, `profile.cufile_stats`.
- `CUFILE_ENV_PATH_JSON` 환경 변수로 실행 시 설정 파일을 지정한다.

### 4.2 측정과 웜업
- 같은 프로세스에서 첫 probe 커널 launch는 65 ms, 두 번째는 0.0232 ms였다. 첫 실행의 준비 비용(지연 모듈 로딩 등 추정)이 측정에 섞인다. 웜업이나 첫 반복 제외가 필요하다. 지연 로딩이 원인인지는 `CUDA_MODULE_LOADING`으로 검증해야 한다 (미확인).
- `posix_memalign`으로 할당한 호스트 버퍼는 첫 접근 때 페이지 폴트가 난다. 첫 `D2H` 복사가 `H2D`보다 느린 이유 중 하나. 측정 전에 버퍼를 한 번 채우는 것이 필요하다.
- `O_DIRECT`는 page cache를 우회하므로 `drop_caches`는 실질적인 효과가 제한적이다.
- 한 커널의 warp 16개가 같은 `%globaltimer` 값을 가졌다. 타이머 해상도가 거칠 가능성이 있어서 warp 간 순서 비교는 신뢰하기 어렵다. 해상도 측정이 필요하다.

### 4.3 계측의 한계
- NVTX는 호스트 타임라인의 구간 표시다. NVTX만으로 GPU 내부 subsystem을 알 수 없다.
- GPU 내부 위치는 SM과 warp slot까지만 PTX 특수 레지스터(`%smid`, `%warpid`)로 얻는다. copy engine, SSD DMA, PCIe root, IOMMU, MMU는 BXTRACE로 보이지 않는다. nsys, Nsight Compute, ftrace/perf, blktrace, PCIe 분석기 등이 필요하다.
- 호스트 시계(`monotonic_us`)와 GPU 시계(`%globaltimer`)는 다른 시계다. 측정 한 번에서는 오프셋이 안정적으로 보였지만 교정 절차는 아직 없다.
- `laneid`는 항상 0이다. lane 0만 기록하기 때문.
- cuFile 내부가 native로 동작했는지 compat로 떨어졌는지는 사용자 코드에서 알 수 없다. `cufile.json`의 `logging.level=INFO`와 log dir 설정으로 로그에서 확인해야 한다.
- 헬퍼 함수 안의 `BXTRACE`는 헬퍼의 줄 번호를 찍는다. 호출 위치를 넘기려면 `BXTRACE_LOC` 또는 `trace_cufile` 매크로 방식이 필요하다.

### 4.4 Spark(GB10)와 GDS
- 사용자 말과 검색 결과 모두 GB10에서 네이티브 GDS는 기대하기 어렵다. Spark의 메모리는 통합 메모리다. 네이티브 경로가 이 구조에서 성립하는지는 실험으로 확인해야 한다.
- 로그 확인 필요: `cufile.json`의 `logging` 설정으로 INFO 수준 로그를 남기고, compat 여부를 기록.

### 4.5 지원 저장장치 (검색으로 확인, 출처 기준)
- NVIDIA 문서 기준 로컬 저장장치는 NVMe가 DMA를 수행해야 한다. 지원 목록은 NVMe, NVMe-oF, 그 위의 EXT4/XFS(ordered 모드), NFS over RDMA, WekaFS, VAST, DDN 등 RDMA 분산 파일시스템.
- SATA HDD와 SATA SSD는 네이티브 GDS 대상이 아니다. compat 모드(POSIX fallback)로만 동작할 가능성이 크다. 단 SATA에서 실제 동작 여부는 검색으로 확인하지 못했다.
- NVMe P2PDMA 모드: 업스트림 NVMe 드라이버만으로 GDS를 쓰는 모드로 릴리스 노트에 언급. 최소 커널 6.2 (Ubuntu 22.04 기준), x86_64 기준. aarch64 GB10에서 적용되는지는 미확인.
- 지원 GPU: 릴리스 노트상 데이터센터와 Quadro급 중 compute capability 기준을 충족하는 카드만 GDS 모드. RTX 3060 사례(포럼)는 compat 전용. RTX PRO 6000이 네이티브 GDS 대상인지는 공식 지원 매트릭스로 확인하지 못했다.

---

## 5. 시나리오 정의 (현재 구현)

| 시나리오 | 저장 대상 | 경로 | 주의 |
|---|---|---|---|
| FS_GDS | 파일시스템 파일 (O_DIRECT) | cuFile | 요청 시나리오 |
| FS_POSIX | 파일시스템 파일 | pwrite/pread + cudaMemcpy | 비교 기준 |
| RAW_GDS | 블록 장치 영역 | cuFile | 영역을 덮어씀. cuFile raw 지원 미검증 |
| RAW_POSIX | 블록 장치 영역 | pwrite/pread | 영역을 덮어씀. 대칭용 추가 |

공통 phase: init, write, read, verify, cleanup. 각 phase는 `Phase` 이벤트로 trace에 표시된다.

raw 시나리오 안전장치: `--raw-device /dev/...` 필수, `--i-accept-overwrite` 필수, 장치 크기 검사, 장치 삭제/truncate 없음.

---

## 6. 미완료 및 미확인 항목

### 6.1 코드 검증
- [ ] `bx_gds_basic.cu` 전체 빌드 (Spark, rtx6000 두 노드 모두). `BX_DEV_PROBE` 매크로, `trace_cufile` 매크로 치환, `<linux/fs.h>`의 `BLKGETSIZE64`, `nvtx3/nvToolsExt.h` 존재 여부.
- [ ] `build.log` 확인 후 오류 수정. 이전 빌드에서는 두 번의 컴파일 오류가 있었음 (필드명, void 반환).
- [ ] 4개 시나리오 각각 실행. 특히 RAW_*는 cuFile 지원 여부부터 확인.
- [ ] `trace_sequence_<SCENARIO>.mmd`를 mermaid 렌더러에서 실제로 렌더링해 문법 확인 (현재 렌더링 미확인).

### 6.2 성능 측정
- [ ] 웜업 추가 (probe 커널 사전 실행, 버퍼 사전 채우기) 후 첫 반복 영향 재측정.
- [ ] 여러 번 반복 (DoE: 5회, 교차 실행).
- [ ] CPU core 고정 (`taskset`).
- [ ] `gpu_elapsed_ms`의 지연 로딩 영향을 `CUDA_MODULE_LOADING`으로 검증.
- [ ] `%globaltimer` 해상도 측정 짧은 test 추가.

### 6.3 모드 판별
- [ ] `cufile.json`에 `logging.level=INFO`, `logging.dir=<run dir>` 적용 후 실제 로그에서 native/compat 판별 기준 찾기.
- [ ] `gdscheck -p` 결과 해석 (p2pdma, nvfs, compat 모드 표시).
- [ ] `/proc/driver/nvidia-fs` 상태 확인 (존재 여부는 env 단계에서 기록됨).

### 6.4 커널 측 계측 (BXTRACE로 불가능한 구성 요소)
- [ ] nsys로 `cudaMemcpy`의 copy engine 활동 확인.
- [ ] nvidia-fs 커널 추적 (ftrace, perf) 가능 여부 확인.
- [ ] NVMe 쪽 (blktrace, iostat, NVMe tracepoint).
- [ ] PCIe ACS, IOMMU 설정 확인 (`lspci -vvv`, 커널 파라미터).
- [ ] 이 결과를 trace 형식으로 합치는 방법 설계 (participant 이름은 `COMPONENTS` 표에 미리 정의됨: CE, NVFS_KMOD, NVME_DRIVER, SSD_DMA, PCIE_ROOT, IOMMU).

### 6.5 보고서
- [ ] 보고서 6장 결과 채우기 (현재 단일 cold-start 실행 결과만 있음).
- [ ] [CITE] 항목을 NVIDIA 공식 문서로 채우기. 현재 검색으로 확인된 URL:
  - https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html (NVMe DMA 요구사항, O_DIRECT)
  - https://docs.nvidia.com/gpudirect-storage/nvidia_fs/index.html (nvidia-fs 지원 목록)
  - https://docs.nvidia.com/gpudirect-storage/release-notes/ (지원 GPU, NVMe P2PDMA)
  - https://github.com/NVIDIA/gds-nvidia-fs (driver 소스)
- [ ] LLM KV cache offload/onboard 절 (4장 analysis) 실제 분석 작성. 블록 크기, batch, 연속성, decode 중첩을 실험으로 검증.

### 6.6 CUDA 코드 주석 요구
- [ ] 사용자 요청은 "각 code line마다 상세한 설명을 comment로". 현재는 블록 단위 주석. 줄 단위 주석 보강 필요.

---

## 7. 다음 작업 순서 (권장)

1. Spark에서 `./bx_gds_run_top.py env` 실행, env 결과 확인 (NVMe 장치, 파일시스템, `gdscheck -p`, `cufile.json`).
2. `./bx_gds_run_top.py build` 실행, `build.log` 오류 수정.
3. FS_POSIX, FS_GDS 1회 실행 후 `cufile.json` 로그에서 compat 여부 확인.
4. 웜업 추가 후 FS_GDS, FS_POSIX 반복 측정.
5. RAW_* 는 전용 영역이 확보된 뒤 진행 (정확한 장치 경로와 오프셋 필요).
6. rtx6000 노드에서 같은 과정 반복, 두 노드 비교.
7. nsys 프로파일링으로 copy engine 활동 확인.
8. 결과를 보고서 6장과 DoE에 반영, [CITE] 채우기.

---

## 8. 새 세션을 시작할 때 전달할 내용

- 이 문서와 `README.md`를 먼저 읽는다.
- 사용자의 응답 선호: 한국어, 결론 먼저, 약어 풀어 쓰기, mermaid, 인용 (URL), 미확인 항목은 명시.
- 파일은 `root/src_pkg/` 아래에만 생성하고, 결과는 `root/exe_out/`에 둔다.
- 검색으로 확인한 사실만 인용하고, 확인하지 못한 것은 "미확인"으로 표시한다.
- 현재 가장 큰 불확실성: (a) CUDA 코드가 아직 컴파일되지 않았다, (b) raw 블록 장치에 대한 cuFile 지원, (c) GB10과 RTX PRO 6000의 네이티브 GDS 동작 여부.
