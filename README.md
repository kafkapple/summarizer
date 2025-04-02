# Summarizer 프로젝트

이 프로젝트는 다양한 소스(YouTube, Pocket, Raindrop.io)에서 콘텐츠를 자동으로 가져오고, 대규모 언어 모델(LLM)을 사용하여 요약한 후, 그 결과를 Notion에 기록하는 자동화 도구입니다.

## 주요 기능

*   **다양한 콘텐츠 소스 지원:** YouTube (플레이리스트/개별 동영상), Pocket (태그별 아티클), Raindrop.io (모든 항목)에서 콘텐츠 가져오기를 지원합니다.
*   **LLM 기반 요약:** LangChain 및 LLM(OpenAI의 GPT 모델 또는 Ollama를 통한 로컬 모델 등)을 활용하여 포괄적인 요약을 생성합니다.
*   **유연한 요약 전략 (Configurable via `config.yaml` -> `summary.strategy`):**
    *   **자동 `stuff` (짧은 텍스트용):**
        *   입력 텍스트의 토큰 수가 미리 계산된 임계값보다 작으면, 설정된 전략과 관계없이 이 전략이 먼저 자동으로 시도됩니다.
        *   전체 텍스트를 단일 프롬프트로 LLM에 한 번만 보내며, 구조화된 JSON 출력(한 문장 요약, 전체 요약, 키워드)을 요청하는 특수 프롬프트를 사용합니다. 짧은 콘텐츠에 효율적입니다.
        *   실패 시 설정 파일에 지정된 전략으로 전환(fallback)됩니다.
    *   **`default` (기본 전략):**
        *   사용자 정의 청킹(chunking), 요약, 병합 프로세스를 사용합니다.
        *   텍스트는 토큰 제한(`chunk_size`, `chunk_overlap`)에 따라 청크로 분할됩니다.
        *   각 청크는 `LLMInterface`를 통해 개별적으로 요약되며, 구조화된 JSON 응답(섹션, 키워드, 청크 요약 등)을 요청합니다.
        *   모든 청크의 구조화된 결과가 병합됩니다. `sections`와 `keywords`는 취합됩니다. 병합된 섹션 정보를 바탕으로 `core_summary`와 `sections_summary`가 생성됩니다.
        *   선택적으로 최종 **정제(refinement) 단계**에서 병합된 컨텍스트(`sections_summary`)를 사용하여 추가 LLM 호출을 시도합니다. 이를 통해 더 일관성 있는 최종 `one_sentence_summary`와 `full_summary` 생성을 시도합니다. 이 정제 호출이 실패하면, 첫 번째 청크의 한 문장 요약을 사용하고 각 청크의 전체 요약을 이어 붙이는 방식으로 전환됩니다.
        *   매우 긴 텍스트는 챕터 기반 접근 방식(`_process_large_text`)으로 처리합니다.
        *   가장 상세하고 구조화된 출력 형식(`one_sentence_summary`, `full_summary`, `core_summary`, `sections_summary`, `keywords`, `sections` 모두 포함)을 제공합니다.
    *   **`map_reduce` (LangChain 표준):**
        *   LangChain의 `load_summarize_chain(chain_type="map_reduce")`를 사용합니다.
        *   각 청크를 독립적으로 요약(Map)한 다음, 이 요약들을 재귀적으로 결합하여 최종 요약을 생성(Reduce)합니다.
        *   **한계점:** 이 체인은 일반적으로 **단일 최종 요약 문자열**을 반환합니다. 현재 구현은 이 문자열을 결과 딕셔너리의 **`full_summary` 필드에만 배치**합니다. `default` 전략과 달리 `one_sentence_summary`, `core_summary`, `sections_summary`, `keywords`, `sections`와 같은 다른 구조화된 필드를 자동으로 생성하지 않습니다 (해당 필드는 비어 있거나 기본값을 가짐).
    *   **`refine` (LangChain 표준):**
        *   LangChain의 `load_summarize_chain(chain_type="refine")`를 사용합니다.
        *   첫 번째 청크를 요약한 다음, 후속 청크와 이전 요약을 함께 처리하여 점진적으로 요약을 개선합니다.
        *   **한계점:** `map_reduce`와 유사하게, 출력은 일반적으로 **단일 최종 요약 문자열**이며, 이는 결과 딕셔너리의 **`full_summary` 필드에만 배치**됩니다. `default` 전략이 제공하는 상세한 구조는 생성되지 않습니다 (다른 필드는 비어 있거나 기본값을 가짐).
    *   **Fallback 메커니즘:** 설정된 전략(`map_reduce`, `refine`)이 실패하면, 안정성을 위해 자동으로 `default` 전략으로 전환됩니다.
*   **구조화된 출력 (특히 `default` 전략 사용 시):** 다음과 같은 다양한 수준의 요약을 생성합니다:
    *   `one_sentence_summary` (한 문장 요약)
    *   `full_summary` (전체 요약)
    *   `core_summary` (핵심 주제)
    *   `sections_summary` (상세 섹션별 요약 내용)
    *   `keywords` (키워드 목록)
    *   `sections` (구조화된 섹션 정보: 제목, 요약 등)
*   **Notion 연동:** 가져온 콘텐츠 정보 및 생성된 요약을 지정된 Notion 데이터베이스에 저장합니다.
*   **설정 관리:** Hydra (`config/config.yaml`)를 사용하여 API 키, Notion ID, LLM 매개변수, 요약 전략 및 기타 설정을 쉽게 구성할 수 있습니다.
*   **로깅:** 디버깅 및 모니터링을 위한 상세 로깅을 제공하며, LLM 상호작용 및 토큰 계산을 위한 별도 로그 파일을 포함합니다.

## 설치

1.  **저장소 복제:**
    ```bash
    git clone <repository-url>
    cd summarizer
    ```
2.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **API 키 및 경로 설정:**
    *   `config/config.yaml.template` 파일이 있다면 `config/config.yaml`로 이름을 변경하거나, 없다면 새로 생성합니다.
    *   `config/config.yaml` 파일에 Notion, OpenAI (또는 Ollama 설정), YouTube, Pocket, Raindrop.io의 API 키를 입력합니다.
    *   올바른 Notion 데이터베이스 ID (`database_id`, 그리고 필요하다면 소스별 ID `pocket_id`, `raindrop_id` 등)를 설정합니다.
    *   YouTube API 접근을 위해서는 Google Cloud Console에서 발급받은 `client_secret.json` 파일이 프로젝트 루트 디렉토리에 필요합니다. `source=youtube`으로 스크립트를 처음 실행하면 OAuth 인증 절차를 안내하며 `token.json` 파일이 생성됩니다.
4.  **요약 설정:**
    *   `config/config.yaml`에서 원하는 `summary.strategy` (`default`, `map_reduce`, `refine`)를 설정합니다. 짧은 텍스트에는 `stuff` 전략이 자동으로 시도됨을 기억하세요.
    *   LLM 매개변수 (`llm.model`, `llm.temperature`, `llm.response_tokens` 등)를 조정합니다.
5.  **Notion 데이터베이스 설정:** 사용하는 Notion 데이터베이스 (`notion.database_id`)가 `config/config.yaml`의 `notion.mapping` 섹션에 정의된 속성 이름 및 타입과 일치하는지 확인합니다. Notion 저장 오류 발생 시 디버그 로그(`--- Properties being sent to Notion ... ---`)를 확인하여 전송되는 정확한 속성을 확인하세요.

## 사용법

프로젝트 루트 디렉토리에서 메인 스크립트를 실행하며, Hydra를 사용하여 설정을 덮어쓸 수 있습니다:

*   **YouTube 플레이리스트 (기본 전략 사용):**
    ```bash
    python main.py source=youtube playlist_url="<your_youtube_playlist_url>"
    ```
*   **Pocket 아티클 (refine 전략 사용):**
    ```bash
    python main.py source=pocket tags=["tag1","tag2"] summary.strategy=refine
    ```
*   **Raindrop.io (map_reduce 전략 사용):**
    ```bash
    python main.py source=raindrop summary.strategy=map_reduce
    ```

**설정 옵션 (`config/config.yaml` 예시):**

*   `source`: `youtube`, `pocket`, `raindrop` 중 선택.
*   `summary.strategy`: `default`, `map_reduce`, `refine` 중 선택.
*   `llm.model`: 사용할 LLM 모델 지정 (예: `gpt-4`, `gpt-3.5-turbo`, 또는 Ollama 모델 이름).
*   `llm.provider`: `openai` 또는 `ollama`.
*   `output.language`: 요약 결과의 대상 언어 (예: `ko`, `en`).
*   `include_keywords`, `enable_chapters` 등: 요약 기능 토글 (주로 `default` 전략의 출력 구조에 영향을 줌).

## 프로젝트 구조

```
summarizer/
├── config/                 # 설정 파일 (Hydra)
│   ├── config.yaml
│   └── ...
├── outputs/
│   ├── logs/                 # 로그 파일
│   └── summaries/            # 저장된 요약 파일 (.json, .md, .txt)
├── src/
│   ├── fetcher/            # 데이터 가져오기 모듈 (youtube.py, pocket.py, ...)
│   ├── logger/             # Notion 로깅 모듈 (base.py, youtube.py, ...)
│   ├── __init__.py
│   ├── lang_sum.py         # 요약기 동기 래퍼
│   ├── llm_interface.py    # 다양한 LLM 상호작용 인터페이스
│   ├── sum.py              # 핵심 요약 로직 (LangChainSummarizer)
│   └── utils.py            # 유틸리티 함수 (로깅, 파일 처리 등)
├── .gitignore
├── main.py                 # 메인 스크립트 진입점
├── README.md
└── requirements.txt        # Python 의존성
```