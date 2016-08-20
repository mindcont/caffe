% 该程序用来 对训练后的模型权值 进行可视化
% 張正軒 (bond@mindcont.com)
% 更多访问  http://blog.mindcont.com
% 来自 《深度学习：21天实战 caffe》

clear;
clc;
close all;
addpath('matlab');
caffe.set_mode_cpu();
caffe.version();
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt','models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel','test')
net.layer_names
net.blob_names
conv1_layer = net.layer_vec(2);
blob1 = conv1_layer.params(1);
w = blob1.get_data();
size(w);
W = zeros(11*3,11*96);
for u = 1:3
    for v = 1:96
        W(11*(u-1) + (1:11),11*(v-1)+(1:11))= w(:,:,u,v);
    end
end

W = W - min(min(W));
W = W / (max(max(W)) * 255);
W = uint8(W);
W = [W,zeros(size(W,1),4*11)];
WW = cat(3,w(1:11,:), W(12:222,:),W(23:33:));
W = zeros(10*12,10*12,3);
for u =1:10
    for v = 1:10
          W((u-1)*12 + (1:11),(v-1)*12 + (1:11),:) = WW(:,(u-1)*11*10 + (v-1)*11 + (1:11), :);
    end
W = uint8(W);
figure;
imshow(W);
