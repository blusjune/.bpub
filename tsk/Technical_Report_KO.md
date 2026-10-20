# task-261020-01 — GDS와 KV Cache I/O: 구조, 소스 추적 및 실험 설계

build: brianmjung_20261020_224500

작성 기준: 2026-10-20. 상태: 문헌·소스 분석 및 계측 도구의 CPU 검증 완료. 사용자 DGX Spark 접속·빌드·추론·SSD DMA 실측은 미수행.

## 1. 결론과 실험 환경에 따른 변경

**GDS(GPUDirect Storage)는 GPU 메모리와 저장장치 사이의 데이터 경로에서 불필요한 CPU 메모리 중간 버퍼를 줄이는 기술입니다. 표준 cuFile 경로의 제어 주체는 CPU입니다.** GPU가 직접 NVMe 명령을 생성한다는 의미는 아닙니다.

**그러나 이번 DGX Spark에서는 native GDS 실험을 전제로 하면 안 됩니다.** NVIDIA 공식 GDS 릴리스 노트의 Known Limitations는 다음과 같이 명시합니다.

> On DGX Spark, GPUDirect Storage is supported only in compatibility mode. Do not load nvidia-fs, because doing so can result in errors.

출처: https://docs.nvidia.com/gpudirect-storage/release-notes/index.html

이는 “cuFile API를 호출할 수 없다”는 뜻이 아닙니다. **cuFile API를 사용하더라도 CPU/system-memory 경유 compatibility 경로이며, native SSD↔GPU P2P DMA 증거가 되지 않는다**는 뜻입니다. Spark에서 API 성공, 높은 대역폭, 또는 작은 CPU 사용률만 보고 native GDS라고 판정해서는 안 됩니다.

| 확인 항목 | 입력/확인 결과 | 해석 |
|---|---|---|
| 사용자 장비 | DGX Spark, GB10, Ubuntu 24.04 | Arm64 및 통합 메모리 플랫폼 |
| 사용자 드라이버 | 580.126.09 | 사용자 제공값; 원격 확인하지 않음 |
| nvidia-smi CUDA Version | 13.0 | 드라이버가 표시하는 CUDA 호환 수준; 설치 nvcc 버전과 동일하다는 보장은 없음 |
| 메모리 표시 | Not Supported | VRAM 크기가 0이라는 의미가 아님 |
| 물리 메모리 | 공식 제품 사양상 128 GB LPDDR5x 통합 메모리 | 분리형 GPU의 HBM과 CPU DRAM으로 모델링하지 않음 |
| native GDS | 공식 문서상 Spark 미지원 | compatibility-only로 분류 |
| uGDS | 확인한 커밋에 무조건적인 x86 pause intrinsic 존재 | Arm64 빌드·P2P 메모리 등록 모두 별도 장애물; 단순 빌드 수정으로 해결된다고 주장할 수 없음 |
| BaM | 프로젝트 요구사항이 PCIe P2P 지원 x86 | Spark에서 바로 실행 가능한 후보로 취급하지 않음 |

Spark의 “통합 메모리”와 CUDA의 `cudaMallocManaged` 할당 방식은 구분해야 합니다. 같은 물리 LPDDR5x를 공유하더라도 할당·매핑·동기화 방식에 따라 추가 복사나 캐시 일관성 비용이 생길 수 있습니다. “통합이므로 모든 복사가 사라진다”도 성립하지 않습니다.

실험은 다음 두 트랙으로 나눕니다.

1. **Spark 트랙:** MHA → GQA → GQA+MoE로 확장하면서 Raw Block KV I/O, 동일 Raw Block 구현의 파일 대조군, cuFile compatibility 경로를 조사합니다.
2. **지원 플랫폼 트랙:** native GDS·uGDS·BaM을 검증한 별도 장비에서 비교합니다. 현재 패키지의 `ugds_raw` 사례는 Spark에서 차단됩니다. 다른 장비가 존재한다고 가정하거나 접속하지 않았습니다.

DGX Spark 하드웨어 출처: https://docs.nvidia.com/dgx/dgx-spark/hardware.html

## 2. 용어와 설계 철학

- **Control plane(제어 경로):** 어떤 토큰의 어느 레이어 KV를 언제 저장/조회할지 결정하고, 버퍼·파일/슬롯·주소·요청·완료를 관리하는 코드입니다.
- **Data plane(데이터 경로):** 실제 KV 바이트가 GPU-visible 메모리, 중간 버퍼, PCIe, SSD 컨트롤러 사이로 이동하는 경로입니다.
- **DMA(Direct Memory Access):** 장치가 CPU의 바이트 단위 복사 루프 없이 메모리를 읽거나 쓰는 방식입니다.
- **P2P(Peer-to-Peer):** PCIe 장치 간 전송을 말합니다. 트랜잭션이 CPU root complex를 지나가는 것과 CPU DRAM에 바이트가 저장되는 것은 다릅니다.
- **NVMe(Non-Volatile Memory Express):** SSD 명령·큐 프로토콜입니다. SQ(Submission Queue), CQ(Completion Queue), MMIO(Memory-Mapped I/O) doorbell을 사용합니다.
- **PRP(Physical Region Page)/SGL(Scatter-Gather List):** NVMe 명령에서 DMA 대상 메모리 구간을 기술하는 방식입니다. GPU 가상 주소를 NVMe 명령에 그대로 넣으면 되는 것이 아닙니다.
- **Offload:** 추론 엔진의 KV를 외부 캐시 계층에 저장합니다. **Onboard**는 여기서 외부 캐시의 KV를 추론 엔진 메모리로 복구하는 의미로 사용합니다. 프로젝트 코드에서는 retrieve/load/prefetch라는 이름이 더 흔합니다.

GDS의 기본 철학은 **계산 엔진이 사용하는 버퍼까지 데이터 경로를 단축하되, CPU·OS의 파일/요청 관리 역할을 유지**하는 것입니다. explicit I/O로 읽을 위치·크기·대상 버퍼를 명시하고, 등록·할당 비용을 재사용하며, CUDA stream 순서로 계산과 I/O 의존성을 연결합니다. 배치·비동기 API도 제어 요청의 CPU 기원을 GPU NVMe 명령 생성으로 바꾸지는 않습니다.

원래 데이터가 CPU에서 파싱되어야 하거나 I/O가 병목이 아니면 이득이 작습니다. 반대로 GPU가 처음 또는 마지막 소비자이고, 큰 정렬된 데이터가 반복 전송되며, 충분한 동시성이 있는 경우 유리합니다. 학습 데이터, 체크포인트, 영상/분석 데이터, GPU 기반 데이터 처리, 재사용 가능한 대형 KV chunk가 대표적인 후보입니다. 4 KiB마다 stream API 한 번을 호출하는 방식이 항상 최선은 아닙니다.

출처: https://docs.nvidia.com/gpudirect-storage/design-guide/index.html
https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html

## 3. GDS · uGDS · BaM 비교

아래 GDS 열은 **로컬 NVMe와 파일시스템을 사용하는 표준 CPU-issued cuFile native 경로**를 기준으로 합니다. 원격 파일시스템·NIC·compatibility mode는 아래 별도 설명을 적용합니다.

| 비교 항목 | NVIDIA GDS/cuFile | uGDS | BaM |
|---|---|---|---|
| 지향점 | 기존 파일/프레임워크와 GPU 직접 데이터 경로 통합 | CPU 사용자 공간의 경량 NVMe 제어 + GPU 직접 데이터 경로 | GPU 계산 중 발생하는 세밀하고 데이터 의존적인 I/O |
| 이름/기원 | GPUDirect Storage | user-space GPU Direct Storage; ScaleX-IO 프로젝트 | Big accelerator Memory; GPU 주도 외부 메모리 접근 연구 |
| 상위 I/O 요청 결정 | CPU의 애플리케이션/프레임워크 | CPU 애플리케이션; 별도 CoPilotIO 통합과 구분 | GPU kernel/thread가 접근 중 결정 |
| NVMe SQE 생성 | CPU에서 실행되는 OS/스토리지 드라이버 계층 | CPU user-space 코드 | GPU device 코드 |
| SQ/CQ 위치 | 일반 로컬 Linux NVMe 경로에서는 호스트 메모리 | 확인한 경로는 CPU 접근 가능한 DMA 큐 | GPU가 접근·관리하는 큐, GPU-side 큐 경로 |
| SQ tail doorbell | CPU의 NVMe driver | CPU user-space MMIO store | GPU-side MMIO store |
| SSD의 명령 소비 | 컨트롤러가 SQE를 가져와 실행 | 동일 | 동일 |
| 데이터 READ | SSD 컨트롤러 DMA가 GPU-visible 대상 메모리에 기록 | 동일 | 동일 |
| 데이터 WRITE | SSD 컨트롤러가 GPU-visible 소스 메모리를 DMA로 읽음 | 동일 | 동일 |
| CQE 작성 | SSD 컨트롤러 | SSD 컨트롤러 | SSD 컨트롤러 |
| 완료 감지 | CPU driver/라이브러리; interrupt/poll 구현에 의존 | CPU busy polling 기본; opt-in interrupt/eventfd 경로 존재 | GPU가 CQ를 관측·처리 |
| CQ head doorbell | CPU driver | CPU user-space 코드 | GPU device 코드 |
| 파일시스템 | 일반 cuFile file API 경로에 존재 | 원시 NVMe namespace 경로; 파일시스템 생략 | 원시 저장 영역을 소프트웨어 캐시가 접근 |
| Linux NVMe data path | 일반 native 파일 경로에서 사용 | bypass; setup용 전용 driver는 여전히 필요 | 일반 파일/NVMe data path bypass; setup/매핑 driver 필요 |
| CPU 역할 | 요청 결정·등록·OS 경로·완료 관리 | 초기화 + 매 I/O 제어 | 초기화·장치 설정·큐/메모리 매핑 등; steady-state data I/O는 GPU 주도 |
| GPU SM 비용 | KV pack/scatter 및 추가 내부 D2D 등이 발생 가능 | KV pack/scatter; I/O 제어는 CPU | I/O 발행·큐 관리·polling·software cache에 SM 자원 사용 |
| 기존 vLLM/LMCache 연결 | GDS backend 및 MP GDS L1 경로 | 확인한 LMCache MP GDS L1 adapter 존재 | 기본 LMCache의 drop-in backend로 확인되지 않음 |
| Spark 판정 | compatibility only | 확인한 source/hardware 경로로는 실험 대상에서 제외 | x86/P2P 요구사항과 불일치 |

**공통점:** 모두 저장 데이터가 GPU 계산에 공급되는 경로를 개선하려고 합니다. native 직접 경로에서는 payload를 SSD 쪽 DMA 엔진이 이동시킵니다. CPU/GPU가 NVMe 명령을 만들었다는 사실과 DMA 엔진이 데이터를 옮긴다는 사실은 별개입니다.

**제어상 핵심 차이:** GDS와 uGDS는 둘 다 CPU 제어지만, uGDS는 CPU 사용자 공간이 SQE·PRP·doorbell·CQ를 관리합니다. BaM은 그 제어 작업까지 GPU kernel로 옮깁니다. “uGDS = GPU-initiated I/O”로 쓰면 틀립니다.

**uGDS와 CoPilotIO:** uGDS는 CoPilotIO의 동기 실험에서 출발한 사용자 공간 라이브러리라고 프로젝트가 설명합니다. CoPilotIO의 GPU-CPU 협업 요청 전달 구조와 라이브러리의 CPU 호출 API를 같은 구현이라고 취급하지 않습니다. 논문 성능 수치를 LMCache adapter의 성능 보장으로 옮길 수도 없습니다.

**원격 GDS:** NIC 또는 스토리지 컨트롤러의 DMA/RDMA가 GPU 메모리에 접근할 수 있습니다. 로컬 SSD가 GPU를 직접 DMA하는 위 표를 모든 원격 배치에 그대로 적용하면 안 됩니다.

**GDS의 driver 변형:** `nvidia-fs` 기반 경로와 지원 플랫폼의 upstream NVMe P2PDMA 경로가 있습니다. 모든 GDS 실행에 `nvidia-fs`가 필수라고 일반화할 수 없습니다. Spark에는 이 일반론보다 compatibility-only 제한이 우선합니다.

출처: https://github.com/ScaleX-IO/uGDS
https://github.com/ZaidQureshi/bam
https://arxiv.org/abs/2203.04910
https://www.usenix.org/conference/osdi26/presentation/chen-guanyi

## 4. KV offload에 GDS를 적용하는 적절한 위치

권장 개념 경로는 **paged KV → GPU상의 연속 staging buffer → storage DMA → SSD**이며, 반대 방향에서는 **SSD → GPU staging → paged KV**입니다. 여기서 “CPU bounce buffer 제거”와 “GPU 내부 복사 제거”는 다른 목표입니다. LMCache가 paged KV를 모아 저장하는 pack/gather 및 복구 scatter는 여전히 GPU에서 필요할 수 있습니다.

1. CPU가 토큰 prefix, 모델·dtype·TP rank·레이어 구분 등을 이용해 캐시 키와 hit 범위를 결정합니다.
2. GPU가 실제 재사용 대상 KV를 pack/gather합니다.
3. 재사용하는 staging buffer를 등록하고, 정렬된 offset/크기로 I/O를 제출합니다.
4. offload에서는 producer kernel 완료 이후에 WRITE가 시작되도록 stream/event 의존성을 둡니다.
5. onboard에서는 READ의 실제 완료·바이트 수를 확인하고 scatter 후 attention 소비가 시작되도록 합니다.
6. buffer/슬롯 참조는 DMA 완료까지 유지합니다. API submit 반환 시점에 재사용하거나 해제하면 안 됩니다.
7. 작은 요청은 배치하고, 큰 chunk는 제한 크기와 등록 구간 경계에 따라 분할합니다.

LLM 추론의 **prefix KV 재사용**과, 매 decode step에 모든 과거 KV를 SSD에서 불러오는 **attention working-set paging**은 다릅니다. 기본 실험은 전자입니다. 같은 prefix가 hit하면 prefill을 생략할 수 있지만, 같은 decode step마다 KV 전체를 SSD에서 다시 읽는다고 가정하지 않습니다. Prefill KV만 저장하는 정책에서는 decode token마다 SSD WRITE가 생기지 않을 수도 있습니다.

직접 경로의 이득은 다음 조건에서 판단합니다.

`회피한 prefill 시간 > lookup + SSD read + staging/scatter + 의존성 대기 비용`

저장 비용은 hit 확률·재사용 횟수·SSD 쓰기 예산까지 포함합니다. Cache hit 비율 자체가 높더라도 CPU L1이나 vLLM 내부 cache hit라면 SSD 효과를 평가한 것이 아닙니다.

## 5. 확인한 소스의 실행 주체와 함수 경로

전체 GitHub 저장소 5개를 실제로 내려받았습니다. 정확한 SHA는 `source_manifest.json`, 함수 시작·종료 줄과 인자는 `function_inventory.csv`에 있습니다. 관련 원본은 `source_evidence/`에 보존했습니다. 이 SHA들은 서로 독립적인 HEAD 스냅샷이며 **검증된 상호 호환 버전 조합은 아닙니다.** 모델 weights는 내려받지 않았으며, 모델 config 3개와 구현 소스는 보존했습니다.

| 코드/동작 | 실행 주체 | 주의점 |
|---|---|---|
| vLLM scheduler, connector, Python model.forward | CPU | model.forward가 GPU 실행 코드라는 뜻은 아님; kernel launch를 수행하는 host 코드 |
| attention, projection, FFN, MoE routing/expert 계산 | GPU kernel | Python 함수 trace만으로 kernel 실행 종료를 알 수 없음 |
| LMCache 키 생성·lookup·slot allocation·RPC·스레드 작업 | CPU | CUDA IPC handle/event도 CPU에서 설정 |
| LMCache `device_ops.multi_layer_block_kv_transfer` | CPU host dispatch → GPU kernel | gather/scatter 실제 바이트 처리는 GPU 측 구현 |
| `gpu_ops.lmcache_memcpy_async_d2h/h2d` | CPU 함수 | 이름이 d2h/h2d여도 `GDSMemoryObject`이면 GDS transfer 분기로 들어감 |
| `GDSContext.transfer_async` | CPU | registered staging 구간마다 I/O 분할·제출 |
| `AsyncHandle.read_async/write_async` | CPU | cuFile/uGDS native API 호출; 반환은 제출 결과 |
| `uGDS async_io_callback → do_io_internal` | CPU callback | 확인한 CUDA 경로는 `cudaLaunchHostFunc` 사용; GPU kernel이 SQ를 만드는 경로가 아님 |
| RawBlock Rust binding, pread/pwrite/io_uring | CPU/kernel | host buffer 기준; `zero_copy` 명칭은 GPU P2P 보장이 아님 |
| NVMe 명령 소비, DMA, CQE 생성, FTL, NAND 접근 | SSD controller | NAND 내부 실행은 host 함수 로그로 직접 관측할 수 없음 |

### 5.1 MP GDS L1 경로

이번 실행 명령은 LMCache 쪽 `lmcache.integration.vllm.lmcache_mp_connector`를 명시합니다. vLLM 저장소 자체의 동명 connector와 무심코 섞지 않습니다.

**저장 진입:** `LMCacheMPConnector.wait_for_save`가 STORE request들을 모아 worker adapter의 `batched_submit_store_requests`로 전달합니다. `save_kv_layer`와 `wait_for_layer_load`라는 callback 이름만 보고, 해당 MP 경로가 항상 레이어 단위 I/O를 수행한다고 결론내리면 안 됩니다. 확인한 구현의 일부 callback은 기본적으로 no-op이고 experimental dispatcher 분기가 있습니다.

**로드 진입:** `LMCacheMPConnector.start_load_kv` → worker adapter `batched_submit_retrieve_requests` → MP 서버 처리·storage manager → object-group transfer입니다.

서버 측 주요 관측 후보:

- `StorageManager.reserve_write`, `finish_write`, `submit_prefetch_task`, `query_prefetch_status`, `read_prefetched_results`, `finish_read_prefetched`.
- `object_group_transfer.transfer_kv_per_object_group`, `run_direct_transfer` 및 내부 transfer plan.
- `device_ops.multi_layer_block_kv_transfer`의 host 호출, `gpu_ops.lmcache_memcpy_async_d2h/h2d`.
- `GDSContext.transfer_async`, `_slab_write`, `_slab_read`, `_record_submission`.
- cuFile 또는 uGDS `AsyncHandle.write_async/read_async`.

이 목록은 **정적 관측 후보**입니다. workload·transfer mode·backend 분기에 따라 일부만 실행됩니다. source function 목록을 한 줄의 강제 call chain으로 꾸며내지 않습니다.

### 5.2 중요한 세 가지 backend 구분

| 경로 | 실제 구조 | 보존 특성 |
|---|---|---|
| MP `raw_block` L2 | GPU KV ↔ CPU L1 ↔ RawBlockCore/Rust ↔ raw device | slot header와 checkpoint에 의한 재시작 복구 |
| MP `--gds-l1-backend cufile` | GPU staging ↔ cuFile ↔ filesystem slab | L1 allocator/index는 휘발성; 초기화 시 파일 truncate/preallocate |
| MP `--gds-l1-backend ugds` | GPU staging ↔ uGDS ↔ raw character device | GDS L1 slab의 휘발성 인덱스; RawBlock L2 checkpoint와 별개 |

GDS L1 사용 시 CPU pinned-DRAM L1이 대체됩니다. raw_block 같은 byte-array L2를 같은 GDS L1에 무조건 결합하지 않습니다. legacy in-process `GdsBackend`는 별도 구현이며, chunk data/metadata 파일 경로를 가집니다. legacy YAML과 MP CLI를 섞지 않습니다.

MP GDSContext에는 GPU 등록 구간 경계(현재 구현 최대 16 MiB)에 따른 분할이 있습니다. 예를 들어 24 MiB의 논리 chunk라도 native 호출 한 번 또는 NVMe 명령 한 번이라고 가정할 수 없습니다. SSD MDTS, PRP/SGL 제한, OS split/merge가 추가로 작용합니다.

출처: https://docs.lmcache.ai/mp/configuration.html
https://docs.lmcache.ai/mp/l2_storage/raw_block.html
https://docs.lmcache.ai/kv_cache/storage_backends/gds.html

## 6. 상세 WRITE/READ sequence

`../diagrams/`의 .mmd는 **소스와 프로토콜을 바탕으로 작성한 설계도**입니다. 실측 trace가 아닙니다. 자동 생성 관측 다이어그램은 실제 JSONL에 기록된 CPU/native 이벤트만 표시합니다.

### 6.1 Filesystem + native cuFile: 지원 플랫폼

초기화: CPU가 slab 파일을 생성/사전할당하고 `O_DIRECT`로 열어 handle을 등록합니다. GPU buffer와 stream을 등록합니다. 준비 비용은 steady-state와 분리합니다.

WRITE:

1. CPU connector가 저장 범위·key를 결정하고 storage manager가 slab offset을 할당합니다.
2. GPU gather가 paged KV를 연속 staging buffer로 모읍니다.
3. producer 의존성 뒤에 CPU가 `cuFileWriteAsync`를 제출합니다.
4. 파일시스템은 file offset을 extent/block으로 해석하고 direct I/O를 준비합니다. page cache payload 우회와 filesystem 우회는 다릅니다.
5. CPU NVMe driver가 command/PRP 또는 SGL을 만들고 SQ를 갱신한 뒤 SQ tail doorbell을 씁니다.
6. SSD는 명령을 가져와 GPU-visible 소스 메모리를 DMA READ하여 WRITE 데이터를 받습니다. 즉 **NVMe WRITE의 payload 전송은 SSD가 메모리를 읽는 방향**입니다.
7. SSD의 FTL/NAND 처리 및 write-cache 정책에 따라 완료 조건이 충족되면 CQE를 씁니다.
8. CPU driver/라이브러리가 CQ를 처리하고 CQ head doorbell을 갱신합니다. stream 완료 및 bytes_done 검증 이후에 buffer/slot을 재사용합니다.

READ는 SSD가 데이터를 읽고 **GPU 메모리에 DMA WRITE**한다는 점이 역방향입니다. READ 완료 → scatter 완료 → attention 소비의 순서를 보장해야 합니다.

NVMe WRITE 완료와 NAND 영속 기록 완료는 항상 동의어가 아닙니다. volatile write cache, power-loss protection, FUA/flush 정책을 별도 기록합니다. GDS는 내구성 프로토콜을 대신하지 않습니다.

### 6.2 Raw Block + uGDS: 별도 지원 플랫폼

CPU가 raw character device를 열어 device capacity를 검증하고, GPU DMA 매핑·큐·slab을 준비합니다. 파일 offset 대신 slab byte offset → namespace LBA로 변환합니다.

`AsyncHandle.write_async` → `uGDSWriteAsync` → `cudaLaunchHostFunc`가 enqueue한 **CPU** `async_io_callback` → `do_io_internal` → 등록된 GPU buffer의 DMA 주소 탐색 → device 제한으로 분할 → `nvm_sq_enqueue` → `nvm_cmd_header`, `nvm_cmd_data_ptr`, `nvm_cmd_rw_blks` → memory fence → `nvm_sq_submit` → SSD DMA → `wait_for_completion` → status 확인 → `nvm_sq_update`/`nvm_cq_update` → bytes_done 기록입니다.

SQ tail/CQ head doorbell은 user-space CPU MMIO store입니다. 초기화 driver가 없어지는 것은 아닙니다. async API가 반환된 직후에는 위 callback이 아직 실행 전일 수 있습니다. 기본 polling 외 interrupt/eventfd 경로도 존재합니다.

계측 patch는 `do_io_internal` 진입/종료, SQ tail doorbell 직전, CQE 관측 직후를 기록합니다. CQE 관측 marker는 status 검사 전이며, 성공을 뜻하지 않습니다. `ugds_batch.cpp`의 별도 batch 경로는 이 patch의 범위가 아닙니다.

### 6.3 Raw Block L2: Spark에서 우선 조사

WRITE 후보 경로:

`GPU gather → GPU/CPU buffer transfer → CPU L1 object → RawBlockL2Adapter.submit_store_task → _run_store_task → RawBlockCore.put_many → _write_one 또는 _put_many_batch_io → _write_buffers → Rust pwrite_from_buffer/io_uring → Linux NVMe driver → SSD`

READ 후보 경로:

`key lookup → submit_load_task → _run_load_task → RawBlockCore.load_many_into → _read_buffers → Rust pread_into/io_uring → CPU L1 destination → GPU staging/scatter → attention`

Spark에서는 위 GPU/CPU buffer가 같은 물리 메모리 풀에 속하더라도 API 수준 복사·동기화가 존재할 수 있으므로 Nsight와 host 메모리 계측으로 확인합니다.

`use_odirect=true`이면 host page-cache payload 경유를 피할 수 있지만, alignment 보정용 bounce buffer가 추가될 수 있습니다. `enable_zero_copy=true`는 가능한 host buffer 경로의 복사를 줄이는 옵션이며 GPU P2P DMA를 선언하는 옵션이 아닙니다.

`io_engine=posix` → `io_uring` → `use_uring_cmd=true`는 후속 단계로 하나씩 바꿉니다. `io_uring_cmd`는 Linux NVMe driver를 완전히 제거하는 uGDS와 같지 않습니다. NVMe passthrough character device와 O_DIRECT block device도 구분합니다.

Raw backend에도 slot header, checkpoint, recovery validation I/O가 있습니다. “filesystem이 없으므로 순수 KV payload만 기록된다”는 가정은 틀립니다.

### 6.4 동일 Raw Block 구현 + XFS 파일 대조군

RawBlockCore에 raw device 대신 **미리 크기를 확보한 일반 파일**을 줍니다. slot/header/checkpoint 로직을 유지하고, filesystem extent 매핑과 metadata 경로를 추가합니다. 이를 raw device와 비교하는 것이 filesystem 영향 분리에 더 적합합니다.

반면 `ugds_raw`와 `cufile_fs` 비교는 파일시스템뿐 아니라 kernel/user I/O stack·threading·API가 함께 바뀝니다. 그 차이를 전부 “XFS overhead”라고 부를 수 없습니다. 동일 raw/file 비교조차 device 분할·LBA 배치·단편화·SSD 상태가 달라질 수 있으므로 paired run, SSD preconditioning, swap 순서 반복이 필요합니다.

## 7. 단계별 DoE와 모델

본문에서 **DoE = Design of Experiments(실험계획법)**입니다. 원 요청의 “GQA 및 DoE 구조”는 문맥상 **GQA 및 MoE(Mixture of Experts)**를 뜻하는 것으로 해석했습니다. 다른 의미라면 모델 확장 단계만 조정하면 됩니다.

| 단계 | 모델 | Attention/FFN | 레이어 / Query heads / KV heads / head dim | FP16/BF16 KV bytes/token | 256-token chunk의 논리 KV |
|---|---|---|---|---:|---:|
| 1 | facebook/opt-125m | MHA + dense | 12 / 12 / 12 / 64 | 36,864 B | 9 MiB |
| 2 | Qwen/Qwen2.5-0.5B | GQA + dense | 24 / 14 / 2 / 64 | 12,288 B | 3 MiB |
| 3 | Qwen/Qwen3-30B-A3B | GQA + MoE | 48 / 32 / 4 / 128 | 98,304 B | 24 MiB |

모델 config는 `models/`에 보존했습니다. 표는 TP=1, K/V 양쪽, 모든 레이어, 2 bytes/element, full attention, 비압축을 가정한 이론량이며 SSD 실측량은 아닙니다.

일반식:

`KV bytes/token = 2 × num_layers × num_KV_heads × head_dim × bytes_per_element`

`logical_chunk_bytes = KV bytes/token × actual_chunk_tokens`

MHA(Multi-Head Attention)는 각 Query head에 별도 K/V를 둡니다. GQA(Grouped-Query Attention)는 여러 Query head가 K/V head를 공유합니다. MoE는 주로 FFN expert 계산 구조를 바꾸며 **expert 수만큼 KV를 곱하지 않습니다.** 동일 모델 geometry에서 MHA→GQA의 KV 비율은 `Hkv/Hq`이지만, 위 서로 다른 모델 전체의 KV 차이를 GQA 효과만으로 해석할 수는 없습니다.

Qwen3-30B-A3B의 active parameter 약 3B를 weight 상주량 3B로 착각해서는 안 됩니다. 전체 약 30B weight는 2 bytes 기준 약 60 GB 수준이며, runtime·KV·workspace·OS 여유가 추가로 필요합니다. Spark에서 실제 가용 메모리를 확인한 뒤 3단계로 진행합니다. 모델들은 공개 weight 모델이며 각 모델 라이선스를 별도로 따릅니다.

### 7.1 Gate와 실험 순서

| Gate | 실험 | 통과 조건 |
|---|---|---|
| 0 | preflight, OS/driver/toolkit/GDS/메모리/장치 확인 | Spark compatibility-only 분류; root SSD를 raw 타깃으로 사용하지 않음 |
| 1 | OPT 기본 추론, LMCache 미연결 대조 | 출력·token 수 정상; 실행 환경 고정 |
| 2 | LMCache CPU L1 연결, store/retrieve 경로 | callback·RPC·storage manager trace와 hit counter 일치 |
| 3 | OPT + Raw Block L2 | filesystem 우회; 실제 L2 WRITE와 READ 완료·payload 검증 |
| 4 | 같은 backend의 pre-sized XFS file | 논리 payload/metadata 정책을 유지하고 block I/O 차이 비교 |
| 5 | cuFile filesystem compatibility | host 경유 경로임을 기록; direct DMA로 분류하지 않음 |
| 6 | GQA → GQA+MoE | geometry로 예상한 KV 바이트와 관측 전송량 대조 |
| 7 | 지원 플랫폼의 native GDS/uGDS | direct-path 증거, 등록·completion·결과 검증 후에만 성능 비교 |
| 8 | MLA/압축/quantized KV | 기본 모델 분석 완료 후 선택적으로 확장 |

**Raw Block을 첫 저장장치 실험으로 사용한다**는 요청을 반영했습니다. 파일은 나중 대조군입니다. GPU 없이 실행한 도구 unit test의 임시 파일은 KV 저장장치 실험 결과가 아닙니다.

### 7.2 요인과 통제

`doe/doe_cases.csv`에는 204개 **계획 사례**가 있습니다. 실행 결과가 아니며 모든 사례는 UNMEASURED입니다. Spark에서 uGDS는 BLOCKED_NATIVE, cuFile은 COMPAT_ONLY로 표시합니다.

- 기본값: TP=1, vLLM block 16 tokens, requested LMCache chunk 256 tokens, concurrency 1, output 32 tokens, prefix caching 비활성, eager mode, decode-cache save 비활성.
- MHA 확장: requested chunk 128/512, input 257/1792 tokens, concurrency 1/4. 257은 chunk/block 경계의 partial tail을 조사합니다. OPT는 input+output ≤2048을 지킵니다.
- 재사용: cold(다른 첫 토큰), full(완전히 같은 prefix), partial(정렬된 prefix까지만 동일) 후보를 만듭니다. prefix가 중간에서 달라지면 이후 chunk content가 같아도 chaining된 key가 달라질 수 있습니다.
- 순서: 같은 stage 내부에서 순서를 난수화하고 3회 독립 프로세스/상태 재설정 반복을 권장합니다. warm-up과 steady-state를 분리하고, 성능 측정은 tracing off/얕은 NVTX/상세 trace 세 수준으로 오버헤드를 평가합니다.
- 차후 요인: io_uring, passthrough, checkpoint, FDP(Flexible Data Placement), 압축, TP, CUDA graphs, layerwise transfer는 한 번에 하나씩 추가합니다.

**실제 chunk 크기를 기록해야 합니다.** 최신 MP connector는 요청 chunk 크기를 geometry 정렬 조건에 맞게 상향 협상할 수 있습니다. CLI의 256이 최종256이라는 가정은 금물입니다. 현재 slot 계산기는 requested chunk를 기준으로 한 초안이므로 negotiation 결과가 다르면 slot size를 다시 계산해야 합니다.

**SSD READ를 유도하는 방법:** 두 번째 같은 요청만 보내서는 부족합니다. CPU L1 용량을 넘는 서로 다른 prefix로 L1을 채우되 L2 working set은 보존하고, 처음 prefix를 다시 읽어 L2 hit/실제 READ를 확인합니다. GDS L1에는 이 CPU L1 우회 문제가 다르게 적용됩니다. vLLM GPU prefix hit, LMCache CPU hit, SSD hit를 각각 분리합니다. 서버를 재시작해서 모든 index를 없애는 방식은 휘발성 GDS L1에서 hit를 보장하지 않습니다.

**인과 추론 제한:** OPT→Qwen 비교에서는 layer/width/context/FFN/커널이 함께 변합니다. attention 방식만의 효과를 추정하려면 같은 geometry에서 Hkv만 바꾼 별도 통제 모델이 필요합니다. MoE 역시 같은 attention geometry의 dense/MoE 통제군을 추가해야 합니다. 본 204개 사례는 경로 이해와 특성화 계획이며 이 통제 실험을 대신하지 않습니다.

### 7.3 수집 지표

| 레벨 | 지표 | 목적 |
|---|---|---|
| 서비스 | TTFT(Time to First Token), TPOT(Time per Output Token), throughput, p50/p95/p99 | I/O가 실제 서비스에 미치는 영향 |
| 논리 KV | hit tokens, chunk/object 수, logical payload bytes, 실제 저장 tail | 알고리즘이 요구한 데이터량 |
| API | offset, size, split 수, enqueue 시간, completion 결과 | framework/library 증폭 |
| OS block | issue/complete, I/O size, LBA, QD, merge/split, latency | filesystem/driver 영향 |
| SSD | host read/write bytes·commands, queue 정책, temperature, sustained BW | 실제 장치 부하 |
| 시스템 | CPU 코어시간·polling 사용률, GPU SM/D2D, shared RAM 사용량, page cache | 숨은 비용 |

계층별로 `API bytes / logical KV bytes`, `block bytes / API bytes`, `SSD host bytes / logical KV bytes`를 따로 계산합니다. 메타데이터·checkpoint·alignment padding·재시도·duplicated store를 구분합니다. NAND WAF(Write Amplification Factor)는 NAND 내부 쓰기량/host 쓰기량이며 host 로그만으로 계산할 수 없습니다.

`bx_workload.py`는 non-streaming end-to-end 시간을 기록하는 pilot입니다. **TTFT 측정기가 아닙니다.** TTFT/TPOT의 정식 측정은 streaming client 또는 vLLM serving benchmark를 사용하고 time origin과 queue time 포함 여부를 고정해야 합니다.

## 8. 계측 도구와 생성 다이어그램의 의미

### 8.1 Python 함수와 NVTX

`bxtrace.py`와 opt-in `sitecustomize.py`가 선택한 소스 경로의 Python 함수 call/return 실행 구간을 JSONL로 기록합니다. timestamp는 같은 host에서 비교 가능한 `monotonic_ns`, 식별자는 PID/TID/span/parent/task, 항목은 function/file/line/인자/반환 요약입니다. 기존 profiler가 있으면 충돌을 피하도록 중단합니다.

- generator/coroutine의 yield·await 전후는 **별도 CPU 실행 segment**입니다. return event가 I/O 완료나 최종 함수 종료를 보장하지 않습니다.
- thread별로 관측하고 worker spawn에 환경이 전파되도록 합니다. profiler 활성화 전 시작된 thread는 누락될 수 있습니다.
- tensor 내용을 CPU로 읽거나 `item()`으로 암묵 동기화하지 않습니다. 복잡한 객체는 type 요약이며 모든 내부 인자를 완전 직렬화하지는 않습니다.
- `BX_NVTX=1`이면 `nvtx` 패키지를 이용한 host range도 남깁니다. NVTX(NVIDIA Tools Extension)는 annotation이며 그 자체가 GPU 실행 trace가 아닙니다.
- 매우 자세한 profiling은 함수마다 JSON 쓰기와 전역 lock 비용이 있습니다. 진단용입니다. 성능 비교는 동일 경로에서 tracing off 대조군을 둡니다.
- 자동 source wrapper로 generator/async semantics를 바꾸지 않고 런타임 계측을 적용했습니다. uGDS에는 별도의 실제 소스 patch를 적용했습니다.

### 8.2 Native source instrumentation

`bx_instrument_ugds.py`는 확인한 `src/ugds_io.cpp`의 정확한 anchor가 1개씩 존재할 때만 source를 수정합니다. `.bx-original` 백업과 `ugds_trace.patch`가 있습니다. 현재 작업 사본에 실제 적용해 anchor 적합성을 확인했습니다. 빌드·실제 NVMe 실행은 하지 않았습니다.

`bx_native_trace.hpp`는 C++17 RAII scope와 marker를 JSONL로 남깁니다. native span은 Python span과 자동으로 연결되지 않습니다. offset/size/시간으로 연결 후보를 찾을 수 있으나 concurrency에서 모호하면 동일 요청으로 단정하지 않습니다. 정확한 cross-process/native 인과 연결에는 request/chunk/operation ID를 RPC와 async request 구조체까지 전달하는 추가 통합이 필요합니다. 이 통합은 구현·실행 검증 범위에 포함되지 않습니다.

cuFile 내부 driver 함수는 이 Python profiler로 보이지 않습니다. block trace, Nsight Systems, 가능하면 libcufile stats를 병행합니다. uGDS native data path는 kernel NVMe/block 경로를 우회하므로 block trace가 비어 있다고 I/O가 없다고 결론내리지 않습니다.

### 8.3 Mermaid 변환기

`bx_gen_seqdiag_from_tracelog.py`는 BX JSONL을 받아 `.md`에 paginated Mermaid sequence를 생성합니다.

- 순서가 바뀐 로그를 timestamp로 정렬하고 PID+span으로 짝을 찾습니다.
- 잘못된 JSON/schema/clock, 중복 입력/span, orphan end, thread 불일치, 미완료 span을 검출합니다.
- `--allow-incomplete`는 종료된 프로세스나 필터링된 로그를 명시적으로 부분 trace로 표시하는 옵션입니다.
- Mermaid/HTML control 문자를 제거하고, synthetic sample은 눈에 띄게 표시합니다.
- 관측하지 않은 GPU/SSD/DMA 화살표나 완료 성공을 자동 생성하지 않습니다.
- 서로 다른 host의 monotonic clock은 직접 합치지 않습니다. 현재 도구에는 분산 clock 보정 기능이 없습니다.
- NVTX/CUPTI/Nsight 파일을 자동 import하거나 kernel event와 합치는 기능은 없습니다. 입력은 본 패키지의 BX JSONL입니다.

CPU 함수 계측·native 일부 지점·개념 시퀀스까지가 이번 구현입니다. **완전한 request→GPU→NVMe 물리 이벤트의 실측 causality trace가 완성됐다는 의미는 아닙니다.**

## 9. Adversarial verification

반례 중심 테스트 코드와 실제 실행 로그는 `tests/`에 저장합니다. 검증한 범위와 남은 범위를 분리합니다.

| 검증 | 결과/판정 |
|---|---|
| coroutine suspend/resume, 동시 task | CPU 실행 segment 분리 및 균형 검증 |
| generator yield/resume | 함수 의미 보존 및 segment 생성 |
| exception 발생 | 애플리케이션 동작 보존; trace return을 성공으로 표시하지 않음 |
| 여러 thread | TID 분리 확인 |
| NaN 인자 | JSON 부적합 수를 문자열로 정규화 |
| input 순서 뒤바뀜 | timestamp 정렬 |
| orphan/missing/duplicate span | 오류로 거부 또는 명시적 partial 허용 |
| PID가 다르고 span 문자열 동일 | 충돌 없이 구분 |
| Mermaid 제어문/HTML 주입 | 제어 문자 제거 |
| drift된 uGDS source | source 쓰기 전 거부 |
| C++17 header | g++ 컴파일 및 실행 JSONL 검증 |
| Spark uGDS command | default platform에서 명령 생성 거부 |
| source pin/CLI/source hash | 고정 SHA·파일 hash·정적 목록 보존 |
| CUDA/LMCache/vLLM 통합 빌드 | 미수행 |
| Spark cuFile compatibility 실행 | 미수행 |
| GPU staging/실제 DMA·오류복구·checksum | 미수행 |
| 실제 SSD 데이터 패턴/성능 수치 | 미측정 |

하드웨어 후속 반례: 잘못된 alignment, 짧은 I/O, bytes_done<0, slot 부족, callback 지연, 중간 종료, checkpoint corruption, stale key, 다른 dtype/TP rank/cache salt 혼용, partial tail, 기존 CPU L1 hit, write-cache 영속성 오해를 각각 주입합니다. 동일 prompt의 온전한 KV 재사용과 prefix가 어긋난 잘못된 hit를 구분해야 합니다. 출력 텍스트가 같다는 것만으로 KV 바이트 무결성을 증명하지 않습니다.

## 10. 참고 자료

1. NVIDIA GDS Release Notes: https://docs.nvidia.com/gpudirect-storage/release-notes/index.html
2. NVIDIA GDS Design Guide: https://docs.nvidia.com/gpudirect-storage/design-guide/index.html
3. NVIDIA cuFile API: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
4. NVIDIA GDS Best Practices: https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html
5. DGX Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
6. LMCache MP configuration: https://docs.lmcache.ai/mp/configuration.html
7. LMCache Raw Block: https://docs.lmcache.ai/mp/l2_storage/raw_block.html
8. LMCache source: https://github.com/LMCache/LMCache
9. vLLM source: https://github.com/vllm-project/vllm
10. uGDS source: https://github.com/ScaleX-IO/uGDS
11. BaM source: https://github.com/ZaidQureshi/bam
12. BaM paper, ASPLOS 2023: https://arxiv.org/abs/2203.04910
13. CoPilotIO, OSDI 2026: https://www.usenix.org/conference/osdi26/presentation/chen-guanyi
14. OPT config: https://huggingface.co/facebook/opt-125m/blob/main/config.json
15. Qwen2.5 config: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
16. Qwen3 MoE config: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json

구현 판단은 다운로드한 고정 SHA의 source를 우선합니다. 최신 웹 문서에는 설치된 CUDA 13.0 이후 버전의 내용도 포함되므로 사용자 장비의 실제 패키지·header·CLI 확인 없이 해당 기능을 가정하지 않습니다.


## 11. 상위 실행 구조와 최종 검증

모든 Bash/Python subtest를 `bx_gds_test_top.sh`와 `bx_gds_test_top.py`에서 호출합니다. validate / preflight / fetch / doe / plan / workload / diagram / instrument / smoke / pilot stage를 제공합니다. pilot은 기본 계획 출력이며 `--execute`에서만 서비스를 시작합니다. `--trace`는 함수 계측을 활성화합니다. 자신이 시작한 process group만 종료하고 결과·로그·관측 시퀀스를 run별로 모읍니다.

최종 CPU 테스트는 **17개 모두 통과**했습니다. 상위 dispatcher, 잘못된 stage, Spark native 경로 차단, 모델별 slot geometry, mock HTTP/server lifecycle을 포함합니다. mock 결과는 실제 GPU/NVMe 결과가 아닙니다. CUDA smoke는 미컴파일이며 dependency matrix와 장비 실행은 미검증입니다.
