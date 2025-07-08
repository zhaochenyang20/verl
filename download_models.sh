#!/bin/bash
set -x

# 创建模型目录
mkdir -p $HOME/models

# 下载主要模型 Qwen/Qwen2-7B-Instruct
echo "正在下载 Qwen/Qwen2-7B-Instruct 模型..."
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir $HOME/models/Qwen2-7B-Instruct

# 下载参考模型（ref model）- GRPO训练需要
echo "正在下载参考模型..."
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir $HOME/models/Qwen2-7B-Instruct-ref

# 检查是否需要数据集
echo "检查数据集..."
if [ ! -f "$HOME/data/gsm8k/train.parquet" ]; then
    echo "警告: 未找到训练数据集 $HOME/data/gsm8k/train.parquet"
    echo "请运行数据预处理脚本:"
    echo "python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k"
fi

if [ ! -f "$HOME/data/gsm8k/test.parquet" ]; then
    echo "警告: 未找到测试数据集 $HOME/data/gsm8k/test.parquet"
    echo "请运行数据预处理脚本:"
    echo "python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k"
fi

echo "模型下载完成！"
echo "下载的模型位置："
echo "- 主模型: $HOME/models/Qwen2-7B-Instruct"
echo "- 参考模型: $HOME/models/Qwen2-7B-Instruct-ref"
echo ""
echo "注意: GRPO训练需要参考模型来计算KL散度损失" 