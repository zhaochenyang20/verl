#!/bin/bash
set -x

echo "=== VeRL GRPO训练环境准备脚本 ==="
echo ""

# 检查huggingface-cli是否可用
if ! command -v huggingface-cli &> /dev/null; then
    echo "错误: 未找到huggingface-cli，请先安装: pip install huggingface_hub"
    exit 1
fi

# 检查Python环境
if ! python3 -c "import torch" 2>/dev/null; then
    echo "错误: 未安装PyTorch，请先安装: pip install torch"
    exit 1
fi

# 检查verl是否已安装
if ! python3 -c "import verl" 2>/dev/null; then
    echo "警告: 未安装verl包，正在安装..."
    pip install -e .
fi

echo "1. 准备数据集..."
./prepare_data.sh

echo ""
echo "2. 下载模型..."
./download_models.sh

echo ""
echo "=== 环境准备完成！ ==="
echo ""
echo "现在可以运行GRPO训练:"
echo "bash examples/grpo_trainer/run_qwen2-7b_seq_balance.sh"
echo ""
echo "或者使用自定义参数:"
echo "bash examples/grpo_trainer/run_qwen2-7b_seq_balance.sh --tp 2" 