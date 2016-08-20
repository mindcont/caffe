% 该程序用来 对训练后的模型权值 进行可视化
% 張正軒 (bond@mindcont.com)
% 更多访问  http://blog.mindcont.com
% 来自 《深度学习：21天实战 caffe》

function [] = visualize_weights(w,s)
h = max(size(w,1),size(w,2)); # kernel size
g = h + s ;% grid size ,larger than kernel size for better visual effects

% normalization for gray scale
w = w -min(min(min(min(w))));
w = w / max(max(max(max(w)))) *255;
w = uint8(w);

W =zeros(g * size(w,3), g * size(w,4));
for u = 1:size(w,3)
    for v = 1:size(w,4)
        W (g*(u-1) + (1:h), g*(v-1) + (1:h)) = w (:,:,u,v)';
    end
end
W = uint8(W);
figure;
imshow (W);
