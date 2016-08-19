name: "LeNet"       // 网络（Net）的名称
layer {             // 定义一个 层（layer）
  name: "mnist"     // 层的名字
  type: "Data"      // 层的类型
  top: "data"       // 层的输出blob有两个：data 和 label1
  top: "label"
  include {
    phase: TRAIN    // 该层参数只在 训练阶段有效
  }
  transform_param {
    scale: 0.00390625 //数据变换使用的数据缩放因子
  }
  data_param {       // 数据层参数
    source: "examples/mnist/mnist_train_lmdb"   // lmdb 的路径
    batch_size: 64   // 批量数据，即 一次读取64 张图片
    backend: LMDB    // 数据格式为 lmdb
  }
}
layer {              // 一个新的数据层，名字也叫 mnist ，输出blob 也是data 和 label 唯一不通过的是这里 仅在测试阶段有效
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST      // 该层参数仅在 测试阶段有效
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {             // 定义一个新的卷积层 conv1 ，输入blob 为 data ，输出 blob 为conv1
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1      // 权值学习速率倍乘因子 ，1 表示与全局参数一直
  }
  param {
    lr_mult: 2     // bias 学习速率倍乘因子 ，是全局的两倍
  }
  convolution_param {  // 卷积计算参数
    num_output: 20     // 输出 feature map 数目为20
    kernel_size: 5     // 卷积核尺寸 5*5
    stride: 1          // 卷积输出跳跃间隔，1 表示连续输出，无跳跃
    weight_filler {
      type: "xavier"   // 权值 使用xavier 填充器
    }
    bias_filler {      // bias 使用常数 填充器 ，默认为 0
      type: "constant"
    }
  }
}
layer {                // 定义新的下采样层pool1  输入blob 为 conv1 ，输出blob 为pool1
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {      // 下采样参数
    pool: MAX          // 使用最大值下采样方法
    kernel_size: 2     // 下采样的窗口尺寸 2*2
    stride: 2          // 下采样输出跳跃间隔 2*2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
