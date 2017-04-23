it's hard to find any document about how to use the trained model of MNIST in caffe project.

so I wrote this.if it helps you. pls give me a star or msg.thanks!

2017-03-22 in China.ShenZhen.

by 9crk

```bash
python mnist_test.py 2.png
```
the output result will like below
```bash
pi@DEEPMIND:~/caffe/examples/mnist/test$ python mnist_test.py 2.png
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0423 20:58:33.004647 68981 _caffe.cpp:122] DEPRECATION WARNING - deprecated use of Python interface
W0423 20:58:33.004709 68981 _caffe.cpp:123] Use this instead (with the named "weights" parameter):
W0423 20:58:33.004717 68981 _caffe.cpp:125] Net('../train/lenet.prototxt', 1, weights='../train/caffemodel/lenet_iter_10000.caffemodel')
I0423 20:58:33.412688 68981 net.cpp:58] Initializing net from parameters:
name: "LeNet"
state {
  phase: TEST
  level: 0
}
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 64
      dim: 1
      dim: 28
      dim: 28
    }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
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
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
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
  name: "prob"
  type: "Softmax"
  bottom: "ip2"
  top: "prob"
}
I0423 20:58:33.413276 68981 layer_factory.hpp:77] Creating layer data
I0423 20:58:33.413303 68981 net.cpp:100] Creating Layer data
I0423 20:58:33.413314 68981 net.cpp:408] data -> data
I0423 20:58:33.413358 68981 net.cpp:150] Setting up data
I0423 20:58:33.413380 68981 net.cpp:157] Top shape: 64 1 28 28 (50176)
I0423 20:58:33.413388 68981 net.cpp:165] Memory required for data: 200704
I0423 20:58:33.413398 68981 layer_factory.hpp:77] Creating layer conv1
I0423 20:58:33.413416 68981 net.cpp:100] Creating Layer conv1
I0423 20:58:33.413426 68981 net.cpp:434] conv1 <- data
I0423 20:58:33.413436 68981 net.cpp:408] conv1 -> conv1
I0423 20:58:33.413872 68981 net.cpp:150] Setting up conv1
I0423 20:58:33.413892 68981 net.cpp:157] Top shape: 64 20 24 24 (737280)
I0423 20:58:33.413899 68981 net.cpp:165] Memory required for data: 3149824
I0423 20:58:33.413920 68981 layer_factory.hpp:77] Creating layer pool1
I0423 20:58:33.413935 68981 net.cpp:100] Creating Layer pool1
I0423 20:58:33.413944 68981 net.cpp:434] pool1 <- conv1
I0423 20:58:33.413954 68981 net.cpp:408] pool1 -> pool1
I0423 20:58:33.413970 68981 net.cpp:150] Setting up pool1
I0423 20:58:33.413982 68981 net.cpp:157] Top shape: 64 20 12 12 (184320)
I0423 20:58:33.413990 68981 net.cpp:165] Memory required for data: 3887104
I0423 20:58:33.414000 68981 layer_factory.hpp:77] Creating layer conv2
I0423 20:58:33.414014 68981 net.cpp:100] Creating Layer conv2
I0423 20:58:33.414022 68981 net.cpp:434] conv2 <- pool1
I0423 20:58:33.414033 68981 net.cpp:408] conv2 -> conv2
I0423 20:58:33.414295 68981 net.cpp:150] Setting up conv2
I0423 20:58:33.414311 68981 net.cpp:157] Top shape: 64 50 8 8 (204800)
I0423 20:58:33.414319 68981 net.cpp:165] Memory required for data: 4706304
I0423 20:58:33.414330 68981 layer_factory.hpp:77] Creating layer pool2
I0423 20:58:33.414345 68981 net.cpp:100] Creating Layer pool2
I0423 20:58:33.414355 68981 net.cpp:434] pool2 <- conv2
I0423 20:58:33.414366 68981 net.cpp:408] pool2 -> pool2
I0423 20:58:33.414381 68981 net.cpp:150] Setting up pool2
I0423 20:58:33.414392 68981 net.cpp:157] Top shape: 64 50 4 4 (51200)
I0423 20:58:33.414399 68981 net.cpp:165] Memory required for data: 4911104
I0423 20:58:33.414407 68981 layer_factory.hpp:77] Creating layer ip1
I0423 20:58:33.414418 68981 net.cpp:100] Creating Layer ip1
I0423 20:58:33.414425 68981 net.cpp:434] ip1 <- pool2
I0423 20:58:33.414436 68981 net.cpp:408] ip1 -> ip1
I0423 20:58:33.418272 68981 net.cpp:150] Setting up ip1
I0423 20:58:33.418288 68981 net.cpp:157] Top shape: 64 500 (32000)
I0423 20:58:33.418298 68981 net.cpp:165] Memory required for data: 5039104
I0423 20:58:33.418309 68981 layer_factory.hpp:77] Creating layer relu1
I0423 20:58:33.418320 68981 net.cpp:100] Creating Layer relu1
I0423 20:58:33.418328 68981 net.cpp:434] relu1 <- ip1
I0423 20:58:33.418339 68981 net.cpp:395] relu1 -> ip1 (in-place)
I0423 20:58:33.418354 68981 net.cpp:150] Setting up relu1
I0423 20:58:33.418362 68981 net.cpp:157] Top shape: 64 500 (32000)
I0423 20:58:33.418368 68981 net.cpp:165] Memory required for data: 5167104
I0423 20:58:33.418375 68981 layer_factory.hpp:77] Creating layer ip2
I0423 20:58:33.418385 68981 net.cpp:100] Creating Layer ip2
I0423 20:58:33.418391 68981 net.cpp:434] ip2 <- ip1
I0423 20:58:33.418402 68981 net.cpp:408] ip2 -> ip2
I0423 20:58:33.418465 68981 net.cpp:150] Setting up ip2
I0423 20:58:33.418475 68981 net.cpp:157] Top shape: 64 10 (640)
I0423 20:58:33.418481 68981 net.cpp:165] Memory required for data: 5169664
I0423 20:58:33.418490 68981 layer_factory.hpp:77] Creating layer prob
I0423 20:58:33.418501 68981 net.cpp:100] Creating Layer prob
I0423 20:58:33.418509 68981 net.cpp:434] prob <- ip2
I0423 20:58:33.418516 68981 net.cpp:408] prob -> prob
I0423 20:58:33.418534 68981 net.cpp:150] Setting up prob
I0423 20:58:33.418543 68981 net.cpp:157] Top shape: 64 10 (640)
I0423 20:58:33.418550 68981 net.cpp:165] Memory required for data: 5172224
I0423 20:58:33.418556 68981 net.cpp:228] prob does not need backward computation.
I0423 20:58:33.418563 68981 net.cpp:228] ip2 does not need backward computation.
I0423 20:58:33.418570 68981 net.cpp:228] relu1 does not need backward computation.
I0423 20:58:33.418576 68981 net.cpp:228] ip1 does not need backward computation.
I0423 20:58:33.418582 68981 net.cpp:228] pool2 does not need backward computation.
I0423 20:58:33.418589 68981 net.cpp:228] conv2 does not need backward computation.
I0423 20:58:33.418596 68981 net.cpp:228] pool1 does not need backward computation.
I0423 20:58:33.418604 68981 net.cpp:228] conv1 does not need backward computation.
I0423 20:58:33.418612 68981 net.cpp:228] data does not need backward computation.
I0423 20:58:33.418617 68981 net.cpp:270] This network produces output prob
I0423 20:58:33.418628 68981 net.cpp:283] Network initialization done.
I0423 20:58:33.422605 68981 net.cpp:761] Ignoring source layer mnist
I0423 20:58:33.422894 68981 net.cpp:761] Ignoring source layer loss
[  1.81631463e-11   7.97432346e-12   1.00000000e+00   2.28452972e-08
   3.50044472e-18   1.42160501e-17   1.56527043e-16   1.19107231e-08
   2.24395222e-08   1.76278783e-11]
2
```
