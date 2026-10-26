# inference_consumer.py's image. CPU-only torch/torch_geometric on demo purposes
#
# mlflow + scipy are listed here even though inference_consumer.py never calls them directly:
# it imports classes from src/models/train_tgn.py, and importing that module runs its top-level
# import mlflow / from scipy.stats import spearmanr too - so this image needs them regardless.
#
# Directory structure under /app mirrors the repo (src/serving/..., src/models/...) - both
# common.py and train_tgn.py compute PROJECT_ROOT as 2 parents above themselves, so this must
# stay 2 levels deep for that math to still land on /app inside the container.
FROM python:3.13-slim

WORKDIR /app

# torch from the CPU-only wheel index - must be its own step: that index hosts only
# torch/torchvision/torchaudio, and --index-url fully replaces PyPI rather than adding
# to it, so torch_geometric can't be resolved in the same invocation
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# torch_geometric from regular PyPI, in its own step - pip sees the CPU torch already installed
# above satisfies its dependency, so it's left alone.
RUN pip install --no-cache-dir torch_geometric

RUN pip install --no-cache-dir \
    mlflow \
    scipy \
    redis \
    "aiokafka>=0.14.0" \
    python-dotenv

COPY src/serving/common.py src/serving/inference_consumer.py src/serving/
COPY src/models/train_tgn.py src/models/

CMD ["python", "src/serving/inference_consumer.py"]