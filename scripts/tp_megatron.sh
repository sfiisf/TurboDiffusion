export PYTHONPATH=turbodiffusion
export CUDA_VISIBLE_DEVICES=0,1
nsys profile --trace=cuda,nvtx,osrt --output=tp_megatron_profile --force-overwrite=true \
    torchrun --nproc_per_node=2 turbodiffusion/inference/tp_megatron.py