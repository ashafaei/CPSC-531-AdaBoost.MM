function label = wp_hog( H, test_data, target)
%WP_BRIEF Summary of this function goes here
%   Detailed explanation goes here

if nargin > 2
    test_data = test_data.gray_images(:, :, target);
end

% distance = zeros(H.m, 1);

hog = vl_hog(im2single(test_data), H.cell_size);
feature = squeeze(hog(H.target(1), H.target(2), :));

feature = sqrt(feature);
normalizer = sqrt(sum(feature.^2))+eps;
feature = feature ./ normalizer;
% 
% feature = H.kernel * feature;
% 
% normalizer = sqrt(sum(feature.^2))+eps;
% feature = feature ./ normalizer;

% for i=1:H.m,
%    response =  H.features(i, :)*feature;
%    
%    distance(i) = response;
%    
% end

distance = H.features * feature;


[~, IDX] = max(distance);

label = H.labels(IDX);

end

