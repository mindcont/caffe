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
./build/tools/caffe.bin test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.cafffemodel -iterations 100
```
```
./build/tools/caffe.bin test -model examples/cifar10/cifar10_quick_solver_lr1.prototxt  -weights examples/cifar10/lcifar10_quick_iter_5000.cafffemodel.h5 -iterations 100
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

#### 3.2 解析日志
cd CAFFE_ROOT
```
./tools/extra/parse_log.sh  
./tools/extra/extract_seconds.py
./tools/extra/plot_training_log.py
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
