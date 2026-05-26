



### Build Number
- PWA 형태의 application을 만들 때에는, 언제나 "build: brianmjung_$(date +%Y%m%d_%H%M%S)" 형태의 Build Number를 build 할 때마다 update 해서 PWA의 화면 우측 상단에 font 8 정도의 글씨로 update 될 수 있도록 해줘


### LLM performance simulator

#### version 1

- 지금부터는 HTML5과 javascript 기반으로 동작하는 single file code를 만들꺼야. LLM 특징에 대한 keyword들을 semicolon으로 구분된 형태로 입력 받아서, hugging face 혹은 github.com 에 올라와 있는 주요 open source LLM 들 중 keyword에 matching되는 LLM들 후보 목록을 제시해서 사용자가 선택할 수 있게 해줘. 선택된 LLM의 latest stable release code를 local directory에 download 받은 후, 그 LLM code를 분석해서, 이 LLM의 전체 및 세부 구조는 어떻게 되는지, inference 수행 동작은 어떤 어떤 단계들을 거쳐서 어떤 흐름으로 수행되는지, 이때 각 수행 단계에서 입력으로 받는 matrix들의 data size와, 수행에 필요한 연산량은 어떻게 되는지를 정량적으로 계산하고, 상세 연산 과정 및 전체 흐름을 이해하기 쉽게 visualization할 수 있는 inference performance bottleneck anayzer를 만들어줘.
- 동작에 문제가 없는지 검증하려면 어떻게 하면 좋을까?
- 제안된 방법대로 검증 진행하고, 수정/개선 사항 반영해줘
- 꼭 하나의 html/javascript file로 분석할 필요는 없어. 그리고 필요하다면 전체 흐름 및 code 분석을 위해 chatgpt 도움을 받거나 이를 제대로 수행할 수 있는 analysis code를 만들어서 진행해줘
- 다행히 나는 DGX Spark machine을 하나 가지고 있어. 큰 규모의 LLM을 구동할 수는 없겠지만 기본적인 matrix 연산 수행해서 어느 정도 크기의 matrix 연산에서 얼만큼의 시간이 걸리는지 계산할 수 있어. 이점을 횔용해서 더 큰 규모의 LLM을 더 큰 규모의 (scale-up and scale-out) GPU system에서 구동하는 경우로 profiling extrapolation을 할 수 있을까? 이것을 하려면 어떤 구조로 실험 계획이 설계되어서 DGX Spark에서 어떻게 실제 실험 수행하고 이 결과 값에서 어떻게 production scale의 LLM inference 성능 병목 예측을 할 수 있을까?
- 지금까지 언급된 내용들을 모두 고려해서 제한된 DGX spark machine을 이용해서 성능계수 실측하고 이로부터 얻어진 값들을 config file 로 저장 후 LLM inference system performance model에서 읽어들여서 주어진 가공의 연산 능력과 memory, storage, network 성능을 가진 system에서 소정의 open source LLM inference 실행 시 예상되는 inference speed 및 이때의 성능 병모구지점을 정량적으로 특정할 수 있도록 하는 performance simulator를 만들어줘

#### version 2

- Local SSD (PCIe Gen6 x4 lane, 32TB, 4KiB Read Latency 100us) 혹은 HBF (3TB/s, 512GB, 4KiB Read Latency 2.5us)에 KV Cache를 저장하고, prefix matching을 통해 match되는 KV Cache Tensor를 (GPU computation 없이) 재활용 한다고 가정하자. 이때 매 Inference Session에서 입력 받은 User Prompt의 일부가 25%, 50%, 75% 등 (configure 가능) 소정의 cache hit ratio에 따라서 (GPU computation 없이) KV Cache가 재활용 된다고 가정하자. User Prompt Input Length도 configuration 통해서 range가 설정 가능한데, range 내의 User Prompt Input Length마다 균등한 분포를 띈다고 가정한다. 이러한 KV cache를 사용하게 되면, 그만큼의 HBM 공간을 차지한다는 측면은 있겠으나, KV tensor 계산을 위한 GPU resource는 그만큼 절약된다는 장점이 있겠다. 또한 MoE routed expert weight들을 HBF에 저장하는 경우와, HBF가 없는 경우 HBM에 저장하는 경우로 나뉠텐데, HBF가 없는 경우에는 MoE routed expert weight들이 차지 하는 공간만큼 HBM에 KV tensor가 저장될 space가 부족하게 될 것이다. 반대로 HBF가 있는 경우에는, 그만큼 MoE routed expert weight들이 HBM이 아니라 HBF에 저장되므로, 그만큼 HBM 공간이 더 생긴다고 볼 수 있겠다. 이런 점들이 LLM inference 속도에 얼만큼 영향을 미치는지 계산 가능하도록 calculator file을 작성해줘.
- 이 simulator의 정합성을 검증하려면 어떻게 하면 좋을까?
- 좋아. 나에게 제안한 방법으로 simulator 정합성을 검증하면서, 실측 작업을 수행하고, 이를 기반으로 analytical model 정합성을 더욱 높이는 일련의 작업들을 수행할 수 있도록 bash 기반의 script을 만들어줘. 물론 이 bash script가 제대로 잘 동작하는지 self test도 부탁해.





