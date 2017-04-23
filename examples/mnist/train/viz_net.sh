python_dir="../../../python/"
cd $python_dir

rankdir="LR"
draw_net="draw_net.py"

proto_file="../examples/mnist/train/lenet_train_test.prototxt"
image_file="../examples/mnist/train/lenet_train_test.png"

python $draw_net $proto_file $image_file --rankdir=$rankdir

proto_file="../examples/mnist/train/lenet.prototxt"
image_file="../examples/mnist/train/lenet.png"

python $draw_net $proto_file $image_file --rankdir=$rankdir
