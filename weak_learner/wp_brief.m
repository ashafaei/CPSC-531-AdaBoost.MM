function label = wp_brief( H, test_data, target)
%WP_BRIEF Summary of this function goes here
%   Detailed explanation goes here

if nargin > 2
    test_data = test_data.blurred(:, :, target);
end

distance = zeros(H.m, 1);

for i=1:H.bits,
   response =  test_data(H.patterns(i, 1), H.patterns(i, 2), :) > test_data(H.patterns(i, 3), H.patterns(i, 4), :);
   
   distance = distance + (response ~= H.responses(:, i));
   
end

[~, IDX] = min(distance);

label = H.labels(IDX);

end

