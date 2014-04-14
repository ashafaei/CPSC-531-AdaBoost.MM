function [label, votes] = predict_adaboostmm( x, alpha_mat, H_matrix, k, weak_predictor )
%PREDICT_ADABOOSTMM Summary of this function goes here
%   Detailed explanation goes here

votes = zeros(1, k);

for i=1:numel(H_matrix)
    prediction = weak_predictor(H_matrix{i}, x);
    votes(prediction) = votes(prediction) + alpha_mat(i);
end

[~, label] = max(votes);

end

