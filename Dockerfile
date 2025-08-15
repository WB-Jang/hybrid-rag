# 1. 베이스 이미지 (CUDA 12.1 + PyTorch)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# 2. 환경 변수 설정
# --- Gemini CLI를 위해 nvm 경로 추가 ---
ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/opt/poetry/bin:/home/appuser/.local/bin:${NVM_DIR}/versions/node/v20.16.0/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    ZENML_ANALYTICS_OPT_IN=false \
    ZENML_DEBUG=true \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# 3. root 권한으로 필수 시스템 패키지 및 Poetry 설치
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -

# 4. non-root 사용자 생성 및 작업 디렉토리 설정
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# 5. 작업 디렉토리 지정
WORKDIR /app

# 6. non-root 사용자로 전환
USER appuser

# <<< Gemini CLI 설치 시작 >>>
# 7. nvm(Node Version Manager), Node.js, Gemini CLI 설치
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install 20 && \
    nvm use 20 && \
    npm install -g @google/gemini-cli
# <<< Gemini CLI 설치 종료 >>>

# 8. 의존성 파일 복사 (빌드 캐시 활용)
COPY --chown=appuser:appuser pyproject.toml poetry.lock* ./

# 9. 나머지 애플리케이션 코드 복사
COPY --chown=appuser:appuser . .