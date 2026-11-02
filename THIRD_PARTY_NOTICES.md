# Third-party notices

## RealPDEBench

`flow_model/cno.py` and the CNO copies in `training_templates/` are adapted from [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench), licensed under **CC BY-NC 4.0**. See `licenses/RealPDEBench-LICENSE.txt`.

## NVIDIA StyleGAN3

Portions of the CNO implementation are derived from NVIDIA StyleGAN3 and are covered by the NVIDIA Source Code License. See `licenses/NVIDIA-StyleGAN3-LICENSE.txt` and the headers in the corresponding source files. Upstream: https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt

## Official data and checkpoint

The training data and the CNO initialization checkpoint come from the [official RealPDE competition release](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data) (CC BY-NC 4.0).

## Runtime

PyTorch, Triton, NumPy, h5py, and CUDA are provided by the Docker base image under their own licenses.
