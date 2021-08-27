# common issues

* [RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1] because the unspecified dimension size -1 can be any value and is ambiguous] when using mmtracking related operators.

Solution: make sure the mmdetection version is v2.9.0. 
Reference: https://github.com/open-mmlab/mmtracking/issues/131#issuecomment-821039886

* When exporting videos: Could not find encoder for codec id 27: Encoder not found

Solution: the ffmpeg library installed does not include h264 support (and other formats).
(you can see `--disabled-libx264` in the output of `ffmpeg`)

To install `libx264` enabled `opencv`, execute the following commands (with conda env activated):

```
conda config --add channels conda-forge 
conda config --set channel_priority strict 
```

Then install opencv `conda install opencv`

If you execute `ffmpeg` in the current environment(with `conda` activated), you should able to find the following tokens in the configuration `--enabled-libx264 --enable-libopenh264`.

Reference: https://github.com/conda-forge/opencv-feedstock/issues/230#issuecomment-626293053

* No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda-10.2'

Check pytorch 

```
python
>>> import torch
>>> torch.cuda.is_available()
```

If the above returns `False`, then try to install gpu version.

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

also check if there is any pytorch-cpu version, and uninstall it.
`pip uninstall pytorch-cpu`, `conda uninstall pytorch-cpu`
