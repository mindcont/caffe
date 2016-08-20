## caffe 常用指令

張正軒 (bond@mindcont.com)  
更多访问  http://blog.mindcont.com

### 1 绘制模型文件
```
cd CAFFE_ROOT  
```
绘制 examples 下各个例子的模型图
* mnist
```
python ./python/draw_net.py ./examples/mnist/lenet_train_test.prototxt ./examples/mnist/lenet_train_test.png
python ./python/draw_net.py ./examples/mnist/lenet.prototxt ./examples/mnist/lenet.png
python ./python/draw_net.py ./examples/mnist/mnist_autoencoder.prototxt ./examples/mnist/mnist_autoencoder.png
```

* siamese
```
python ./python/draw_net.py ./examples/siamese/mnist_siamese.prototxt ./examples/siamese/mnist_siamese.png
python ./python/draw_net.py ./examples/siamese/mnist_siamese_train_test.prototxt ./examples/siamese/mnist_siamese_train_test.png
```

* net_surgery
```
python ./python/draw_net.py ./examples/net_surgery/bvlc_caffenet_full_conv.prototxt ./examples/net_surgery/bvlc_caffenet_full_conv.png
```

* hdf5_classification
```
python ./python/draw_net.py ./examples/hdf5_classification/nonlinear_auto_test.prototxt ./examples/hdf5_classification/nonlinear_auto_test.png
python ./python/draw_net.py ./examples/hdf5_classification/nonlinear_auto_train.prototxt ./examples/hdf5_classification/nonlinear_auto_train.png
python ./python/draw_net.py ./examples/hdf5_classification/nonlinear_train_val.prototxt ./examples/hdf5_classification/nonlinear_train_val.png
python ./python/draw_net.py ./examples/hdf5_classification/train_val.prototxt ./examples/hdf5_classification/train_val.png
```

* finetune_pascal_detection
```
python ./python/draw_net.py ./examples/finetune_pascal_detection/pascal_finetune_trainval_test.prototxt ./examples/finetune_pascal_detection/pascal_finetune_trainval_test.png
```

* feature_extraction
```
python ./python/draw_net.py ./examples/feature_extraction/imagenet_val.prototxt ./examples/feature_extraction/imagenet_val.png
```

* cifar10
```
python ./python/draw_net.py ./examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt ./examples/cifar10/cifar10_full_sigmoid_train_test_bn.png
python ./python/draw_net.py ./examples/cifar10/cifar10_full.prototxt ./examples/cifar10/cifar10_full.png
python ./python/draw_net.py ./examples/cifar10/cifar10_quick_train_test.prototxt ./examples/cifar10/cifar10_quick_train_test.png
python ./python/draw_net.py ./examples/cifar10/cifar10_quick.prototxt ./examples/cifar10/cifar10_quick.png
```

绘制 models 文件夹下各个模型图
* bvlc_alexnet
```
python ./python/draw_net.py ./models/bvlc_alexnet/deploy.prototxt ./models/bvlc_alexnet/deploy.png
python ./python/draw_net.py ./models/bvlc_alexnet/train_val.prototxt ./models/bvlc_alexnet/train_val.png
```

* bvlc_googlelenet
```
python ./python/draw_net.py ./models/bvlc_googlenet/deploy.prototxt ./models/bvlc_googlenet/deploy.png
python ./python/draw_net.py ./models/bvlc_googlenet/train_val.prototxt ./models/bvlc_googlenet/train_val.png
```

* bvlc_reference_caffenet
```
python ./python/draw_net.py ./models/bvlc_reference_caffenet/deploy.prototxt ./models/bvlc_reference_caffenet/deploy.png
python ./python/draw_net.py ./models/bvlc_reference_caffenet/train_val.prototxt ./models/bvlc_reference_caffenet/train_val.png
```

* bvlc_reference_rcnn_ilsvrc13
```
python ./python/draw_net.py ./models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt ./models/bvlc_reference_rcnn_ilsvrc13/deploy.png
```

* finetune_flickr_style
```
python ./python/draw_net.py ./models/finetune_flickr_style/deploy.prototxt ./models/finetune_flickr_style/deploy.png
python ./python/draw_net.py ./models/finetune_flickr_style/train_val.prototxt ./models/finetune_flickr_style/train_val.png
```

### 2 训练好的模型对数据进行预测

cd CAFFE_ROOT
```
./build/tools/caffe.bin test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -iterations 100
```
```
./build/tools/caffe.bin test -model examples/cifar10/cifar10_quick_solver_lr1.prototxt  -weights examples/cifar10/lcifar10_quick_iter_5000.caffemodel.h5 -iterations 100
```
### 3 解析训练日志

#### 3.1 记录日志
```
#!/usr/bin/env sh
LOG=examples/mnist/log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee -a $LOG
```
```
run
./examples/mnist/train_lenet.sh
```
而另一中方法 将标准输出重定向到log 文件，并放入后台运行
```
./examples/cifar10/train_quick.sh >& cifar.log &
```
.& 表示所有的标准输出 （stdout）和标准错误输出（stderr）重定向到后log的保存文件 cifar.log 。 最后的 & 表示命令后台运行

连续观测log文件的更新
```
tail -f cifar.log
```
#### 3.2 解析日志
cd CAFFE_ROOT
```
./tools/extra/parse_log.sh  
./tools/extra/extract_seconds.py
./tools/extra/plot_training_log.py
```
提取log文件中的loss 值
```
cat cifar.log | grep "train net output " | awk '{print $11}'
```
### 4 绘图
```
cd CAFFE_ROOT
./tools/extra/plot_training_log.py
```
```
Notes:  
    1. Supporting multiple logs.  
    2. Log file name must end with the lower-cased ".log".  
Supported chart types:  
    0: Test accuracy  vs. Iters  
    1: Test accuracy  vs. Seconds  
    2: Test loss  vs. Iters  
    3: Test loss  vs. Seconds  
    4: Train learning rate  vs. Iters  
    5: Train learning rate  vs. Seconds  
    6: Train loss  vs. Iters  
    7: Train loss  vs. Seconds  
```
run

```
python ./tools/extra/plot_training_log.py 0 ./examples/mnist/log/result.png ./examples/mnist/log/train-2016-08-03-15-50-24.log
```

### 5 特征提取
caffe 提供的使用工具 build/tools/extract_features.bin 实现特征提取功能。该程序需要一个训练好的
网络和一个数据输入层，运行后可得到相应数据通过网络某各中间层产生的特征图并保存在磁盘。
```
Usage: extract_features  pretrained_net_param  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type  [CPU/GPU] [DEVICE_ID=0]

Note: you can extract multiple features in one pass by specifying multiple feature blob names and dataset names separated by ','. The names cannot contain white space characters and the number of blobs and datasets must be equal.
```
中文解释
```
extract_features \ //可执行的程序
pretrained_net_param \  //预训练的网络 *.caffemodel
feature_extraction_proto_file \  // 网络描述文件 *.prototxt
extract_feature_blob_name1[,name2,...] \  // 需要提取的 Blob 名称
save_feature_dataset_name1[,name2,...] \ //保存特征名
num_mini_batches \ //做特征提取的数据批量数据
db_type \ //输入数据的格式 lmdb /leveldb
[CPU/GPU] [DEVICE_ID=0] \ // 使用 cpu 还是 gpu ，若 gpu 提供设备编号
```
例子
```
./build/tools/extract_features.bin \
models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
models/bvlc_reference_caffenet/train_val.prototxt \
fc6,fc7,fc8 \
fc6,fc7,fc8 \
10 \
lmdb \
GPU \
0
```

### 6 caffe计时
```
./build/tools/caffe.bin time \
-model examples/mnist/lenet_train_test.prototxt
```
输出结果：
```
pi@DeepMind:~/caffe$ ./build/tools/caffe.bin time \
> -model examples/mnist/lenet_train_test.prototxt
I0820 09:52:13.517204  4075 caffe.cpp:347] Use CPU.
I0820 09:52:13.666025  4075 net.cpp:322] The NetState phase (0) differed from the phase (1) specified by a rule in layer mnist
I0820 09:52:13.666060  4075 net.cpp:322] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I0820 09:52:13.666136  4075 net.cpp:58] Initializing net from parameters:
name: "LeNet"
state {
  phase: TRAIN
  level: 0
  stage: ""
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
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
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0820 09:52:13.666235  4075 layer_factory.hpp:77] Creating layer mnist
I0820 09:52:13.667006  4075 net.cpp:100] Creating Layer mnist
I0820 09:52:13.667021  4075 net.cpp:408] mnist -> data
I0820 09:52:13.667047  4075 net.cpp:408] mnist -> label
I0820 09:52:13.703961  4081 db_lmdb.cpp:35] Opened lmdb examples/mnist/mnist_train_lmdb
I0820 09:52:13.712002  4075 data_layer.cpp:41] output data size: 64,1,28,28
I0820 09:52:13.712213  4075 net.cpp:150] Setting up mnist
I0820 09:52:13.712232  4075 net.cpp:157] Top shape: 64 1 28 28 (50176)
I0820 09:52:13.712239  4075 net.cpp:157] Top shape: 64 (64)
I0820 09:52:13.712242  4075 net.cpp:165] Memory required for data: 200960
I0820 09:52:13.712251  4075 layer_factory.hpp:77] Creating layer conv1
I0820 09:52:13.712276  4075 net.cpp:100] Creating Layer conv1
I0820 09:52:13.712282  4075 net.cpp:434] conv1 <- data
I0820 09:52:13.712296  4075 net.cpp:408] conv1 -> conv1
I0820 09:52:13.712997  4082 blocking_queue.cpp:50] Waiting for data
I0820 09:52:13.874382  4075 net.cpp:150] Setting up conv1
I0820 09:52:13.874418  4075 net.cpp:157] Top shape: 64 20 24 24 (737280)
I0820 09:52:13.874423  4075 net.cpp:165] Memory required for data: 3150080
I0820 09:52:13.874447  4075 layer_factory.hpp:77] Creating layer pool1
I0820 09:52:13.874466  4075 net.cpp:100] Creating Layer pool1
I0820 09:52:13.874471  4075 net.cpp:434] pool1 <- conv1
I0820 09:52:13.874480  4075 net.cpp:408] pool1 -> pool1
I0820 09:52:13.874501  4075 net.cpp:150] Setting up pool1
I0820 09:52:13.874507  4075 net.cpp:157] Top shape: 64 20 12 12 (184320)
I0820 09:52:13.874511  4075 net.cpp:165] Memory required for data: 3887360
I0820 09:52:13.874517  4075 layer_factory.hpp:77] Creating layer conv2
I0820 09:52:13.874529  4075 net.cpp:100] Creating Layer conv2
I0820 09:52:13.874534  4075 net.cpp:434] conv2 <- pool1
I0820 09:52:13.874569  4075 net.cpp:408] conv2 -> conv2
I0820 09:52:13.890202  4075 net.cpp:150] Setting up conv2
I0820 09:52:13.890233  4075 net.cpp:157] Top shape: 64 50 8 8 (204800)
I0820 09:52:13.890239  4075 net.cpp:165] Memory required for data: 4706560
I0820 09:52:13.890257  4075 layer_factory.hpp:77] Creating layer pool2
I0820 09:52:13.890274  4075 net.cpp:100] Creating Layer pool2
I0820 09:52:13.890280  4075 net.cpp:434] pool2 <- conv2
I0820 09:52:13.890291  4075 net.cpp:408] pool2 -> pool2
I0820 09:52:13.890310  4075 net.cpp:150] Setting up pool2
I0820 09:52:13.890316  4075 net.cpp:157] Top shape: 64 50 4 4 (51200)
I0820 09:52:13.890321  4075 net.cpp:165] Memory required for data: 4911360
I0820 09:52:13.890326  4075 layer_factory.hpp:77] Creating layer ip1
I0820 09:52:13.890343  4075 net.cpp:100] Creating Layer ip1
I0820 09:52:13.890348  4075 net.cpp:434] ip1 <- pool2
I0820 09:52:13.890355  4075 net.cpp:408] ip1 -> ip1
I0820 09:52:13.893347  4075 net.cpp:150] Setting up ip1
I0820 09:52:13.893360  4075 net.cpp:157] Top shape: 64 500 (32000)
I0820 09:52:13.893365  4075 net.cpp:165] Memory required for data: 5039360
I0820 09:52:13.893378  4075 layer_factory.hpp:77] Creating layer relu1
I0820 09:52:13.893388  4075 net.cpp:100] Creating Layer relu1
I0820 09:52:13.893391  4075 net.cpp:434] relu1 <- ip1
I0820 09:52:13.893398  4075 net.cpp:395] relu1 -> ip1 (in-place)
I0820 09:52:13.893724  4075 net.cpp:150] Setting up relu1
I0820 09:52:13.893736  4075 net.cpp:157] Top shape: 64 500 (32000)
I0820 09:52:13.893739  4075 net.cpp:165] Memory required for data: 5167360
I0820 09:52:13.893744  4075 layer_factory.hpp:77] Creating layer ip2
I0820 09:52:13.893754  4075 net.cpp:100] Creating Layer ip2
I0820 09:52:13.893757  4075 net.cpp:434] ip2 <- ip1
I0820 09:52:13.893765  4075 net.cpp:408] ip2 -> ip2
I0820 09:52:13.893821  4075 net.cpp:150] Setting up ip2
I0820 09:52:13.893826  4075 net.cpp:157] Top shape: 64 10 (640)
I0820 09:52:13.893829  4075 net.cpp:165] Memory required for data: 5169920
I0820 09:52:13.893836  4075 layer_factory.hpp:77] Creating layer loss
I0820 09:52:13.893843  4075 net.cpp:100] Creating Layer loss
I0820 09:52:13.893847  4075 net.cpp:434] loss <- ip2
I0820 09:52:13.893852  4075 net.cpp:434] loss <- label
I0820 09:52:13.893862  4075 net.cpp:408] loss -> loss
I0820 09:52:13.893882  4075 layer_factory.hpp:77] Creating layer loss
I0820 09:52:13.894022  4075 net.cpp:150] Setting up loss
I0820 09:52:13.894033  4075 net.cpp:157] Top shape: (1)
I0820 09:52:13.894038  4075 net.cpp:160]     with loss weight 1
I0820 09:52:13.894055  4075 net.cpp:165] Memory required for data: 5169924
I0820 09:52:13.894062  4075 net.cpp:226] loss needs backward computation.
I0820 09:52:13.894068  4075 net.cpp:226] ip2 needs backward computation.
I0820 09:52:13.894073  4075 net.cpp:226] relu1 needs backward computation.
I0820 09:52:13.894076  4075 net.cpp:226] ip1 needs backward computation.
I0820 09:52:13.894083  4075 net.cpp:226] pool2 needs backward computation.
I0820 09:52:13.894088  4075 net.cpp:226] conv2 needs backward computation.
I0820 09:52:13.894094  4075 net.cpp:226] pool1 needs backward computation.
I0820 09:52:13.894099  4075 net.cpp:226] conv1 needs backward computation.
I0820 09:52:13.894104  4075 net.cpp:228] mnist does not need backward computation.
I0820 09:52:13.894109  4075 net.cpp:270] This network produces output loss
I0820 09:52:13.894119  4075 net.cpp:283] Network initialization done.
I0820 09:52:13.894173  4075 caffe.cpp:355] Performing Forward
I0820 09:52:14.555274  4075 caffe.cpp:360] Initial loss: 2.34903
I0820 09:52:14.555317  4075 caffe.cpp:361] Performing Backward
I0820 09:52:14.588186  4075 caffe.cpp:369] *** Benchmark begins ***
I0820 09:52:14.588218  4075 caffe.cpp:370] Testing for 50 iterations.
I0820 09:52:14.622347  4075 caffe.cpp:398] Iteration: 1 forward-backward time: 34 ms.
I0820 09:52:14.656512  4075 caffe.cpp:398] Iteration: 2 forward-backward time: 34 ms.
I0820 09:52:14.690943  4075 caffe.cpp:398] Iteration: 3 forward-backward time: 34 ms.
I0820 09:52:14.719223  4075 caffe.cpp:398] Iteration: 4 forward-backward time: 28 ms.
I0820 09:52:14.743410  4075 caffe.cpp:398] Iteration: 5 forward-backward time: 24 ms.
I0820 09:52:14.767426  4075 caffe.cpp:398] Iteration: 6 forward-backward time: 23 ms.
I0820 09:52:14.790845  4075 caffe.cpp:398] Iteration: 7 forward-backward time: 23 ms.
I0820 09:52:14.814903  4075 caffe.cpp:398] Iteration: 8 forward-backward time: 24 ms.
I0820 09:52:14.839207  4075 caffe.cpp:398] Iteration: 9 forward-backward time: 24 ms.
I0820 09:52:14.863078  4075 caffe.cpp:398] Iteration: 10 forward-backward time: 23 ms.
I0820 09:52:14.886373  4075 caffe.cpp:398] Iteration: 11 forward-backward time: 23 ms.
I0820 09:52:14.910437  4075 caffe.cpp:398] Iteration: 12 forward-backward time: 24 ms.
I0820 09:52:14.934537  4075 caffe.cpp:398] Iteration: 13 forward-backward time: 24 ms.
I0820 09:52:14.958601  4075 caffe.cpp:398] Iteration: 14 forward-backward time: 24 ms.
I0820 09:52:14.982657  4075 caffe.cpp:398] Iteration: 15 forward-backward time: 24 ms.
I0820 09:52:15.005842  4075 caffe.cpp:398] Iteration: 16 forward-backward time: 23 ms.
I0820 09:52:15.029911  4075 caffe.cpp:398] Iteration: 17 forward-backward time: 24 ms.
I0820 09:52:15.054055  4075 caffe.cpp:398] Iteration: 18 forward-backward time: 24 ms.
I0820 09:52:15.077878  4075 caffe.cpp:398] Iteration: 19 forward-backward time: 23 ms.
I0820 09:52:15.101485  4075 caffe.cpp:398] Iteration: 20 forward-backward time: 23 ms.
I0820 09:52:15.124990  4075 caffe.cpp:398] Iteration: 21 forward-backward time: 23 ms.
I0820 09:52:15.150055  4075 caffe.cpp:398] Iteration: 22 forward-backward time: 25 ms.
I0820 09:52:15.174142  4075 caffe.cpp:398] Iteration: 23 forward-backward time: 24 ms.
I0820 09:52:15.198202  4075 caffe.cpp:398] Iteration: 24 forward-backward time: 24 ms.
I0820 09:52:15.221789  4075 caffe.cpp:398] Iteration: 25 forward-backward time: 23 ms.
I0820 09:52:15.245331  4075 caffe.cpp:398] Iteration: 26 forward-backward time: 23 ms.
I0820 09:52:15.269268  4075 caffe.cpp:398] Iteration: 27 forward-backward time: 23 ms.
I0820 09:52:15.293318  4075 caffe.cpp:398] Iteration: 28 forward-backward time: 24 ms.
I0820 09:52:15.317324  4075 caffe.cpp:398] Iteration: 29 forward-backward time: 23 ms.
I0820 09:52:15.340711  4075 caffe.cpp:398] Iteration: 30 forward-backward time: 23 ms.
I0820 09:52:15.364962  4075 caffe.cpp:398] Iteration: 31 forward-backward time: 24 ms.
I0820 09:52:15.389387  4075 caffe.cpp:398] Iteration: 32 forward-backward time: 24 ms.
I0820 09:52:15.413470  4075 caffe.cpp:398] Iteration: 33 forward-backward time: 24 ms.
I0820 09:52:15.437470  4075 caffe.cpp:398] Iteration: 34 forward-backward time: 23 ms.
I0820 09:52:15.461086  4075 caffe.cpp:398] Iteration: 35 forward-backward time: 23 ms.
I0820 09:52:15.485317  4075 caffe.cpp:398] Iteration: 36 forward-backward time: 24 ms.
I0820 09:52:15.509759  4075 caffe.cpp:398] Iteration: 37 forward-backward time: 24 ms.
I0820 09:52:15.533807  4075 caffe.cpp:398] Iteration: 38 forward-backward time: 24 ms.
I0820 09:52:15.557339  4075 caffe.cpp:398] Iteration: 39 forward-backward time: 23 ms.
I0820 09:52:15.581601  4075 caffe.cpp:398] Iteration: 40 forward-backward time: 24 ms.
I0820 09:52:15.605835  4075 caffe.cpp:398] Iteration: 41 forward-backward time: 24 ms.
I0820 09:52:15.629952  4075 caffe.cpp:398] Iteration: 42 forward-backward time: 24 ms.
I0820 09:52:15.654170  4075 caffe.cpp:398] Iteration: 43 forward-backward time: 24 ms.
I0820 09:52:15.677590  4075 caffe.cpp:398] Iteration: 44 forward-backward time: 23 ms.
I0820 09:52:15.701509  4075 caffe.cpp:398] Iteration: 45 forward-backward time: 23 ms.
I0820 09:52:15.725483  4075 caffe.cpp:398] Iteration: 46 forward-backward time: 23 ms.
I0820 09:52:15.749532  4075 caffe.cpp:398] Iteration: 47 forward-backward time: 24 ms.
I0820 09:52:15.773211  4075 caffe.cpp:398] Iteration: 48 forward-backward time: 23 ms.
I0820 09:52:15.796720  4075 caffe.cpp:398] Iteration: 49 forward-backward time: 23 ms.
I0820 09:52:15.820911  4075 caffe.cpp:398] Iteration: 50 forward-backward time: 24 ms.
// 对每个层进行计时
I0820 09:52:15.820946  4075 caffe.cpp:401] Average time per layer:
I0820 09:52:15.820991  4075 caffe.cpp:404]      mnist	forward: 0.05546 ms.
I0820 09:52:15.821002  4075 caffe.cpp:407]      mnist	backward: 0.00124 ms.
I0820 09:52:15.821008  4075 caffe.cpp:404]      conv1	forward: 1.62752 ms.
I0820 09:52:15.821015  4075 caffe.cpp:407]      conv1	backward: 1.70178 ms.
I0820 09:52:15.821022  4075 caffe.cpp:404]      pool1	forward: 3.52992 ms.
I0820 09:52:15.821029  4075 caffe.cpp:407]      pool1	backward: 1.84768 ms.
I0820 09:52:15.821036  4075 caffe.cpp:404]      conv2	forward: 4.01876 ms.
I0820 09:52:15.821043  4075 caffe.cpp:407]      conv2	backward: 7.17916 ms.
I0820 09:52:15.821051  4075 caffe.cpp:404]      pool2	forward: 1.85262 ms.
I0820 09:52:15.821058  4075 caffe.cpp:407]      pool2	backward: 1.13688 ms.
I0820 09:52:15.821064  4075 caffe.cpp:404]        ip1	forward: 0.52474 ms.
I0820 09:52:15.821072  4075 caffe.cpp:407]        ip1	backward: 0.88008 ms.
I0820 09:52:15.821079  4075 caffe.cpp:404]      relu1	forward: 0.03218 ms.
I0820 09:52:15.821086  4075 caffe.cpp:407]      relu1	backward: 0.03644 ms.
I0820 09:52:15.821094  4075 caffe.cpp:404]        ip2	forward: 0.02466 ms.
I0820 09:52:15.821101  4075 caffe.cpp:407]        ip2	backward: 0.03574 ms.
I0820 09:52:15.821108  4075 caffe.cpp:404]       loss	forward: 0.07028 ms.
I0820 09:52:15.821115  4075 caffe.cpp:407]       loss	backward: 0.0035 ms.
// 平均前向传播时间
I0820 09:52:15.821127  4075 caffe.cpp:412] Average Forward pass: 11.7583 ms.
// 平均反向传播时间
I0820 09:52:15.821135  4075 caffe.cpp:414] Average Backward pass: 12.8473 ms.
// 平均前向 + 反向传播时间
I0820 09:52:15.821142  4075 caffe.cpp:416] Average Forward-Backward: 24.64 ms.
// 50 次迭代总时间
I0820 09:52:15.821151  4075 caffe.cpp:418] Total Time: 1232 ms.
I0820 09:52:15.821157  4075 caffe.cpp:419] *** Benchmark ends ***
pi@DeepMind:~/caffe$
```
使用 gpu 进行计时
```
./build/tools/caffe.bin time \
-model examples/mnist/lenet_train_test.prototxt \
-gpu 0
```
输出结果
```
pi@DeepMind:/usr/local/cuda/samples/7_CUDALibraries$ cd ~/caffe
pi@DeepMind:~/caffe$ ./build/tools/caffe.bin time \
> -model examples/mnist/lenet_train_test.prototxt \
> -gpu 0
I0820 10:01:00.201431  4117 caffe.cpp:343] Use GPU with device ID 0
I0820 10:01:00.346184  4117 net.cpp:322] The NetState phase (0) differed from the phase (1) specified by a rule in layer mnist
I0820 10:01:00.346220  4117 net.cpp:322] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I0820 10:01:00.346297  4117 net.cpp:58] Initializing net from parameters:
name: "LeNet"
state {
  phase: TRAIN
  level: 0
  stage: ""
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
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
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0820 10:01:00.346393  4117 layer_factory.hpp:77] Creating layer mnist
I0820 10:01:00.346940  4117 net.cpp:100] Creating Layer mnist
I0820 10:01:00.346956  4117 net.cpp:408] mnist -> data
I0820 10:01:00.346989  4117 net.cpp:408] mnist -> label
I0820 10:01:00.347657  4123 db_lmdb.cpp:35] Opened lmdb examples/mnist/mnist_train_lmdb
I0820 10:01:00.354173  4117 data_layer.cpp:41] output data size: 64,1,28,28
I0820 10:01:00.354951  4117 net.cpp:150] Setting up mnist
I0820 10:01:00.354980  4117 net.cpp:157] Top shape: 64 1 28 28 (50176)
I0820 10:01:00.354989  4117 net.cpp:157] Top shape: 64 (64)
I0820 10:01:00.355001  4117 net.cpp:165] Memory required for data: 200960
I0820 10:01:00.355013  4117 layer_factory.hpp:77] Creating layer conv1
I0820 10:01:00.355043  4117 net.cpp:100] Creating Layer conv1
I0820 10:01:00.355051  4117 net.cpp:434] conv1 <- data
I0820 10:01:00.355067  4117 net.cpp:408] conv1 -> conv1
I0820 10:01:00.511482  4117 net.cpp:150] Setting up conv1
I0820 10:01:00.511523  4117 net.cpp:157] Top shape: 64 20 24 24 (737280)
I0820 10:01:00.511529  4117 net.cpp:165] Memory required for data: 3150080
I0820 10:01:00.511556  4117 layer_factory.hpp:77] Creating layer pool1
I0820 10:01:00.511575  4117 net.cpp:100] Creating Layer pool1
I0820 10:01:00.511582  4117 net.cpp:434] pool1 <- conv1
I0820 10:01:00.511592  4117 net.cpp:408] pool1 -> pool1
I0820 10:01:00.511636  4117 net.cpp:150] Setting up pool1
I0820 10:01:00.511643  4117 net.cpp:157] Top shape: 64 20 12 12 (184320)
I0820 10:01:00.511648  4117 net.cpp:165] Memory required for data: 3887360
I0820 10:01:00.511653  4117 layer_factory.hpp:77] Creating layer conv2
I0820 10:01:00.511667  4117 net.cpp:100] Creating Layer conv2
I0820 10:01:00.511672  4117 net.cpp:434] conv2 <- pool1
I0820 10:01:00.511679  4117 net.cpp:408] conv2 -> conv2
I0820 10:01:00.512604  4117 net.cpp:150] Setting up conv2
I0820 10:01:00.512619  4117 net.cpp:157] Top shape: 64 50 8 8 (204800)
I0820 10:01:00.512625  4117 net.cpp:165] Memory required for data: 4706560
I0820 10:01:00.512635  4117 layer_factory.hpp:77] Creating layer pool2
I0820 10:01:00.512645  4117 net.cpp:100] Creating Layer pool2
I0820 10:01:00.512650  4117 net.cpp:434] pool2 <- conv2
I0820 10:01:00.512658  4117 net.cpp:408] pool2 -> pool2
I0820 10:01:00.512691  4117 net.cpp:150] Setting up pool2
I0820 10:01:00.512697  4117 net.cpp:157] Top shape: 64 50 4 4 (51200)
I0820 10:01:00.512701  4117 net.cpp:165] Memory required for data: 4911360
I0820 10:01:00.512703  4117 layer_factory.hpp:77] Creating layer ip1
I0820 10:01:00.512714  4117 net.cpp:100] Creating Layer ip1
I0820 10:01:00.512718  4117 net.cpp:434] ip1 <- pool2
I0820 10:01:00.512724  4117 net.cpp:408] ip1 -> ip1
I0820 10:01:00.515940  4117 net.cpp:150] Setting up ip1
I0820 10:01:00.515959  4117 net.cpp:157] Top shape: 64 500 (32000)
I0820 10:01:00.515962  4117 net.cpp:165] Memory required for data: 5039360
I0820 10:01:00.515974  4117 layer_factory.hpp:77] Creating layer relu1
I0820 10:01:00.515983  4117 net.cpp:100] Creating Layer relu1
I0820 10:01:00.515988  4117 net.cpp:434] relu1 <- ip1
I0820 10:01:00.515995  4117 net.cpp:395] relu1 -> ip1 (in-place)
I0820 10:01:00.516273  4117 net.cpp:150] Setting up relu1
I0820 10:01:00.516285  4117 net.cpp:157] Top shape: 64 500 (32000)
I0820 10:01:00.516293  4117 net.cpp:165] Memory required for data: 5167360
I0820 10:01:00.516299  4117 layer_factory.hpp:77] Creating layer ip2
I0820 10:01:00.516307  4117 net.cpp:100] Creating Layer ip2
I0820 10:01:00.516314  4117 net.cpp:434] ip2 <- ip1
I0820 10:01:00.516321  4117 net.cpp:408] ip2 -> ip2
I0820 10:01:00.516777  4117 net.cpp:150] Setting up ip2
I0820 10:01:00.516790  4117 net.cpp:157] Top shape: 64 10 (640)
I0820 10:01:00.516796  4117 net.cpp:165] Memory required for data: 5169920
I0820 10:01:00.516805  4117 layer_factory.hpp:77] Creating layer loss
I0820 10:01:00.516816  4117 net.cpp:100] Creating Layer loss
I0820 10:01:00.516821  4117 net.cpp:434] loss <- ip2
I0820 10:01:00.516829  4117 net.cpp:434] loss <- label
I0820 10:01:00.516836  4117 net.cpp:408] loss -> loss
I0820 10:01:00.516860  4117 layer_factory.hpp:77] Creating layer loss
I0820 10:01:00.517065  4117 net.cpp:150] Setting up loss
I0820 10:01:00.517076  4117 net.cpp:157] Top shape: (1)
I0820 10:01:00.517081  4117 net.cpp:160]     with loss weight 1
I0820 10:01:00.517097  4117 net.cpp:165] Memory required for data: 5169924
I0820 10:01:00.517103  4117 net.cpp:226] loss needs backward computation.
I0820 10:01:00.517109  4117 net.cpp:226] ip2 needs backward computation.
I0820 10:01:00.517114  4117 net.cpp:226] relu1 needs backward computation.
I0820 10:01:00.517118  4117 net.cpp:226] ip1 needs backward computation.
I0820 10:01:00.517123  4117 net.cpp:226] pool2 needs backward computation.
I0820 10:01:00.517129  4117 net.cpp:226] conv2 needs backward computation.
I0820 10:01:00.517135  4117 net.cpp:226] pool1 needs backward computation.
I0820 10:01:00.517140  4117 net.cpp:226] conv1 needs backward computation.
I0820 10:01:00.517145  4117 net.cpp:228] mnist does not need backward computation.
I0820 10:01:00.517150  4117 net.cpp:270] This network produces output loss
I0820 10:01:00.517160  4117 net.cpp:283] Network initialization done.
I0820 10:01:00.517216  4117 caffe.cpp:355] Performing Forward
I0820 10:01:00.554848  4117 caffe.cpp:360] Initial loss: 2.31234
I0820 10:01:00.554883  4117 caffe.cpp:361] Performing Backward
I0820 10:01:00.579291  4117 caffe.cpp:369] *** Benchmark begins ***
I0820 10:01:00.579319  4117 caffe.cpp:370] Testing for 50 iterations.
I0820 10:01:00.585279  4117 caffe.cpp:398] Iteration: 1 forward-backward time: 5.42512 ms.
I0820 10:01:00.590579  4117 caffe.cpp:398] Iteration: 2 forward-backward time: 5.26979 ms.
I0820 10:01:00.595906  4117 caffe.cpp:398] Iteration: 3 forward-backward time: 5.30202 ms.
I0820 10:01:00.601218  4117 caffe.cpp:398] Iteration: 4 forward-backward time: 5.28342 ms.
I0820 10:01:00.608657  4117 caffe.cpp:398] Iteration: 5 forward-backward time: 7.39741 ms.
I0820 10:01:00.614018  4117 caffe.cpp:398] Iteration: 6 forward-backward time: 5.28861 ms.
I0820 10:01:00.619390  4117 caffe.cpp:398] Iteration: 7 forward-backward time: 5.3449 ms.
I0820 10:01:00.624630  4117 caffe.cpp:398] Iteration: 8 forward-backward time: 5.21603 ms.
I0820 10:01:00.629947  4117 caffe.cpp:398] Iteration: 9 forward-backward time: 5.29395 ms.
I0820 10:01:00.637393  4117 caffe.cpp:398] Iteration: 10 forward-backward time: 7.41613 ms.
I0820 10:01:00.642751  4117 caffe.cpp:398] Iteration: 11 forward-backward time: 5.31421 ms.
I0820 10:01:00.648030  4117 caffe.cpp:398] Iteration: 12 forward-backward time: 5.25139 ms.
I0820 10:01:00.653406  4117 caffe.cpp:398] Iteration: 13 forward-backward time: 5.34934 ms.
I0820 10:01:00.658699  4117 caffe.cpp:398] Iteration: 14 forward-backward time: 5.26746 ms.
I0820 10:01:00.663983  4117 caffe.cpp:398] Iteration: 15 forward-backward time: 5.25299 ms.
I0820 10:01:00.671483  4117 caffe.cpp:398] Iteration: 16 forward-backward time: 7.45024 ms.
I0820 10:01:00.676774  4117 caffe.cpp:398] Iteration: 17 forward-backward time: 5.25069 ms.
I0820 10:01:00.682129  4117 caffe.cpp:398] Iteration: 18 forward-backward time: 5.32858 ms.
I0820 10:01:00.687453  4117 caffe.cpp:398] Iteration: 19 forward-backward time: 5.29802 ms.
I0820 10:01:00.692703  4117 caffe.cpp:398] Iteration: 20 forward-backward time: 5.22464 ms.
I0820 10:01:00.699779  4117 caffe.cpp:398] Iteration: 21 forward-backward time: 7.04243 ms.
I0820 10:01:00.705302  4117 caffe.cpp:398] Iteration: 22 forward-backward time: 5.47158 ms.
I0820 10:01:00.710268  4117 caffe.cpp:398] Iteration: 23 forward-backward time: 4.93568 ms.
I0820 10:01:00.715337  4117 caffe.cpp:398] Iteration: 24 forward-backward time: 5.04464 ms.
I0820 10:01:00.720330  4117 caffe.cpp:398] Iteration: 25 forward-backward time: 4.97014 ms.
I0820 10:01:00.725335  4117 caffe.cpp:398] Iteration: 26 forward-backward time: 4.97619 ms.
I0820 10:01:00.732550  4117 caffe.cpp:398] Iteration: 27 forward-backward time: 7.17264 ms.
I0820 10:01:00.737597  4117 caffe.cpp:398] Iteration: 28 forward-backward time: 5.00829 ms.
I0820 10:01:00.742584  4117 caffe.cpp:398] Iteration: 29 forward-backward time: 4.96208 ms.
I0820 10:01:00.747560  4117 caffe.cpp:398] Iteration: 30 forward-backward time: 4.9527 ms.
I0820 10:01:00.752538  4117 caffe.cpp:398] Iteration: 31 forward-backward time: 4.95408 ms.
I0820 10:01:00.759523  4117 caffe.cpp:398] Iteration: 32 forward-backward time: 6.95328 ms.
I0820 10:01:00.764935  4117 caffe.cpp:398] Iteration: 33 forward-backward time: 5.3623 ms.
I0820 10:01:00.769999  4117 caffe.cpp:398] Iteration: 34 forward-backward time: 5.03146 ms.
I0820 10:01:00.774997  4117 caffe.cpp:398] Iteration: 35 forward-backward time: 4.97104 ms.
I0820 10:01:00.779958  4117 caffe.cpp:398] Iteration: 36 forward-backward time: 4.93392 ms.
I0820 10:01:00.784934  4117 caffe.cpp:398] Iteration: 37 forward-backward time: 4.94762 ms.
I0820 10:01:00.792230  4117 caffe.cpp:398] Iteration: 38 forward-backward time: 7.26691 ms.
I0820 10:01:00.797286  4117 caffe.cpp:398] Iteration: 39 forward-backward time: 5.01181 ms.
I0820 10:01:00.802366  4117 caffe.cpp:398] Iteration: 40 forward-backward time: 5.04947 ms.
I0820 10:01:00.807343  4117 caffe.cpp:398] Iteration: 41 forward-backward time: 4.94746 ms.
I0820 10:01:00.812299  4117 caffe.cpp:398] Iteration: 42 forward-backward time: 4.9271 ms.
I0820 10:01:00.818864  4117 caffe.cpp:398] Iteration: 43 forward-backward time: 6.53235 ms.
I0820 10:01:00.824565  4117 caffe.cpp:398] Iteration: 44 forward-backward time: 5.65766 ms.
I0820 10:01:00.829633  4117 caffe.cpp:398] Iteration: 45 forward-backward time: 5.03389 ms.
I0820 10:01:00.834741  4117 caffe.cpp:398] Iteration: 46 forward-backward time: 5.07843 ms.
I0820 10:01:00.839735  4117 caffe.cpp:398] Iteration: 47 forward-backward time: 4.96666 ms.
I0820 10:01:00.844683  4117 caffe.cpp:398] Iteration: 48 forward-backward time: 4.92147 ms.
I0820 10:01:00.851516  4117 caffe.cpp:398] Iteration: 49 forward-backward time: 6.79734 ms.
I0820 10:01:00.857131  4117 caffe.cpp:398] Iteration: 50 forward-backward time: 5.5319 ms.
I0820 10:01:00.857151  4117 caffe.cpp:401] Average time per layer:
I0820 10:01:00.857154  4117 caffe.cpp:404]      mnist	forward: 0.0475021 ms.
I0820 10:01:00.857161  4117 caffe.cpp:407]      mnist	backward: 0.00165632 ms.
I0820 10:01:00.857167  4117 caffe.cpp:404]      conv1	forward: 0.242765 ms.
I0820 10:01:00.857173  4117 caffe.cpp:407]      conv1	backward: 0.561014 ms.
I0820 10:01:00.857179  4117 caffe.cpp:404]      pool1	forward: 0.10066 ms.
I0820 10:01:00.857187  4117 caffe.cpp:407]      pool1	backward: 0.502104 ms.
I0820 10:01:00.857192  4117 caffe.cpp:404]      conv2	forward: 0.338195 ms.
I0820 10:01:00.857199  4117 caffe.cpp:407]      conv2	backward: 2.43568 ms.
I0820 10:01:00.857208  4117 caffe.cpp:404]      pool2	forward: 0.0344762 ms.
I0820 10:01:00.857219  4117 caffe.cpp:407]      pool2	backward: 0.157612 ms.
I0820 10:01:00.857228  4117 caffe.cpp:404]        ip1	forward: 0.196381 ms.
I0820 10:01:00.857234  4117 caffe.cpp:407]        ip1	backward: 0.232468 ms.
I0820 10:01:00.857237  4117 caffe.cpp:404]      relu1	forward: 0.0132115 ms.
I0820 10:01:00.857240  4117 caffe.cpp:407]      relu1	backward: 0.0127616 ms.
I0820 10:01:00.857245  4117 caffe.cpp:404]        ip2	forward: 0.0514003 ms.
I0820 10:01:00.857252  4117 caffe.cpp:407]        ip2	backward: 0.0464224 ms.
I0820 10:01:00.857260  4117 caffe.cpp:404]       loss	forward: 0.103868 ms.
I0820 10:01:00.857267  4117 caffe.cpp:407]       loss	backward: 0.0348832 ms.
I0820 10:01:00.857285  4117 caffe.cpp:412] Average Forward pass: 1.27361 ms.
I0820 10:01:00.857292  4117 caffe.cpp:414] Average Backward pass: 4.22636 ms.
I0820 10:01:00.857300  4117 caffe.cpp:416] Average Forward-Backward: 5.54864 ms.
I0820 10:01:00.857309  4117 caffe.cpp:418] Total Time: 277.432 ms.
I0820 10:01:00.857318  4117 caffe.cpp:419] *** Benchmark ends ***
```
* K40 关闭ECC 并开启最大时钟频率
```
# 关闭ecc
sudo nvidia-smi -i 0 --ecc-config=0 # 对每个gpu 重复 -i x

# 重启
sudo reboot

# 设置gpu 模式 为 persistence
sudo nvidia-smi -pm 1

# 设置时钟速率
sudo nvidia-smi -i 0 -ac ****
```
**注意** 此设置每当驱动重新加载/重启 都会复位 。建议将上述命令加入 /etc/rc.local

### 可视化

* 数据可视化
mnist
详情见 /data/mnist/show_mnist_data.m
cifar10
详情见 /data/cifar10/show_cifar10_data.m

* 模型可视化

* 权值可视化
