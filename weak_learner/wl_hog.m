function H = wl_hog( train_data, cost_matrix )
%WL_HOG Summary of this function goes here
%   Detailed explanation goes here

images = train_data.gray_images;
m = size(train_data.blurred, 3);

cell_size = 8;     %randi([4 8]);

target    = randi([1 floor(32/cell_size)], [1 2]);

features  = zeros(m, 31);

for i=1:m,
   hog = vl_hog(im2single(images(:, :, i)), cell_size);

   features(i, :) = hog(target(1), target(2), :);
    
end

features = sqrt(features);

normalizer = sqrt(sum(features.^2, 2))+eps;
features = features ./ repmat( normalizer, 1, 31);

% K=50;
% 
% kernel = 1./sqrt(K) *randn(K, 31);
% 
% features = (kernel * features')';
% 
% normalizer = sqrt(sum(features.^2, 2))+eps;
% features = features ./ repmat( normalizer, 1, K);

[features, IDX] = datasample(features, randi([100 800]));

H = struct();
H.features  = features;
H.target    = target;
H.labels    = train_data.labels(IDX);
H.cell_size = cell_size;
H.m         = m;
% H.kernel    = kernel;

end

