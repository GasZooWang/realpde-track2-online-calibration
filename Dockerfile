FROM w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4
RUN python -c "import sys,torch,numpy,h5py; assert sys.version_info[:2] == (3,10); assert torch.__version__.split('+')[0] == '2.2.2'; assert torch.version.cuda == '12.1'; assert numpy.__version__ == '1.26.4'; assert h5py.__version__ == '3.16.0'"
COPY . /workspace
RUN python -c "import solution; assert callable(solution.train); assert callable(solution.get_ttt_model)"
