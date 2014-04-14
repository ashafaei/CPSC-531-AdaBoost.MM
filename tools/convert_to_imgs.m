function [ output_images ] = convert_to_imgs( input_rows )
%CONVERT_TO_IMGS Summary of this function goes here
%   Detailed explanation goes here

img_count = size(input_rows, 1);

output_images = zeros([32, 32, 3, img_count], 'uint8');

for i=1:img_count,
    output_images(:, :, 1, i) = reshape(input_rows(i, 1:1024), 32, 32)';
    output_images(:, :, 2, i) = reshape(input_rows(i, 1025:2048), 32, 32)';
    output_images(:, :, 3, i) = reshape(input_rows(i, 2049:3072), 32, 32)';
end

end

