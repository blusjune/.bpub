## Requests

- Nvidia DGX Spark에서 chatbot service, vibe coding을 최고의 품질로 수행할 수 있도록 하는 Open Source LLM을 추천해줘.

- Claude Code와 가장 유사한 coding agent 환경을 구축하려면?

- OpenCode, Ollama는 무엇인가? 어떤 역할? 둘 간의 차이점? vLLM, LMCache와의 역할 차이?

- OpenCode + vLLM + Qwen + LMCache 조합으로 시작해볼께. 이것들을 binary package 말고 source code build 방식으로 AI coding 환경을 구축하려고 해.  이 작업들을 위한 root directory는 /opt/work/ai_coding/root/ 이고 필요한 source code 들은 github.com이나  hugging face 에서 git clone 등의 방식으로 download 할 예정이야 (source code의 root directory는 /opt/work/ai_coding/root/src/ ). 이 작업들을 수행하는 top level bash shell script (e.g., bx_ai_coding_setup.sh)을 만들고, 실행 sequence number를 포함하는 sub command (e.g., 010_conf_env, 020_download_src, and so on)를 통해서 세부 작업 진행할 수 있도록 해줘.





