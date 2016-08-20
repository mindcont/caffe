% 该程序用来 对训练后的模型权值 进行可视化
% 張正軒 (bond@mindcont.com)
% 更多访问  http://blog.mindcont.com
% 来自 《深度学习：21天实战 caffe》

clear;
clc;
close all;
addpath('matlab');
caffe.setmode_cpu();
fprintf(['caffe version = ', caffe.version(),'\n'] );

net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt','models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel' ,'test');


fprintf('load net done Net layers :');
net.layer_name;

fprintf('net blobs:');
net.blob_names;

% conv1 weights visualization
conv1_layer = net.layer_vec(2);
blob1 = conv1_layer.params(1);
w = blob1.get_data();
fprintf('conv1 weights shape :');
size(w);
visualize_weights(w,1);

% conv2 weights visualization
conv2_layer = net.layer_vec(6);
blob2 = conv2_layer.params(1);
w2 = blob2.get_data();
fprintf('conv2 weights shape :');
size(w2);
visualize_weights(w2,1);

% conv3 weights visualization
conv3_layer = net.layer_vec(10);
blob3 = conv3_layer.params(1);
w3 = blob3.get_data();
fprintf('conv3 weights shape :');
size(w3);
visualize_weights(w3,1);


% conv4 weights visualization
conv4_layer = net.layer_vec(12);
blob4 = conv4_layer.params(1);
w4 = blob4.get_data();
fprintf('conv4 weights shape :');
size(w4);
visualize_weights(w4,1);

% conv5 weights visualization
conv5_layer = net.layer_vec(14);
blob5 = conv5_layer.params(1);
w5 = blob5.get_data();
fprintf('conv5 weights shape :');
size(w5);
visualize_weights(w5,1);
