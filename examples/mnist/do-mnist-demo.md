### how to do mnist demo



```bash
cd Caffe/data/mnist

# get mnist dataset
./get_mnist.sh

# visual ubyte
mkdir train
python visual_mnist_data.py

# create lmdb
cd caffe_root
 ./examples/mnist/create_mnist.sh

# train with lenet
cd ../../examples/mnist/train
./viz_net.sh


```
