FROM pytorch/torchserve:0.3.0-gpu 

# Cuda version
ARG CUV=cu110

RUN pip install torch==1.7.1+$CUV torchvision==0.8.2+$CUV torchaudio===0.7.2 -f \
    https://download.pytorch.org/whl/torch_stable.html 
RUN pip install wheel
RUN python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/$CUV/torch1.7/index.html

CMD /bin/bash