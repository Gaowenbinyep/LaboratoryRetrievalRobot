export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

CUDA_VISIBLE_DEVICES=0 \
vllm serve /media/a822/82403B14403B0E83/Gwb/WechatRobot/Base_models/Qwen3-1.7B \
    --tensor-parallel-size 1 \
    --port 8888 \
    --disable-log-requests \
    > ./logs/model_output.log 2>&1 &
