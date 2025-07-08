#!/bin/bash
set -x

# 创建数据目录
mkdir -p $HOME/data

echo "正在准备GSM8K数据集..."

# 检查是否已安装verl
if ! python3 -c "import verl" 2>/dev/null; then
    echo "错误: 未安装verl包，请先安装: pip install -e ."
    exit 1
fi

# 运行GSM8K数据预处理
echo "运行GSM8K数据预处理..."
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

echo "数据预处理完成！"
echo "数据集位置: $HOME/data/gsm8k/"
echo "- 训练集: $HOME/data/gsm8k/train.parquet"
echo "- 测试集: $HOME/data/gsm8k/test.parquet" 