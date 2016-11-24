#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
# 该脚本 将 MINST 原始数据集 转换为 可供caffe 调用的 lmdb / leveldb 格式
# 取决于  $BACKEND 所对应的值
set -e

# lmdb/leveldb 生成路径
EXAMPLE=examples/mnist
# 原始数据集路径
DATA=data/mnist
# 二进制文件路径
BUILD=build/examples/mnist

# 后端类型
BACKEND="lmdb"

echo "Creating ${BACKEND}..."

# 如果已经存在 lmdb/leveldb,则先删除
rm -rf $EXAMPLE/mnist_train_${BACKEND}
rm -rf $EXAMPLE/mnist_test_${BACKEND}

# 创建训练数据集
$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_${BACKEND} --backend=${BACKEND}
# 创建测试数据集
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
