function adaboost_out = learn_adaboostmm( adaboost )
%LEARN_ADABOOSTMM Summary of this function goes here
%   Detailed explanation goes here

adaboost_out = struct();

m = adaboost.n_examples;
k = adaboost.n_classes;

f_matrix = zeros(m, k);
C_matrix = zeros(m, k);

H_matrix = cell(adaboost.T, 1);

alpha_mat= zeros(adaboost.T, 1);

if isfield(adaboost, 'watcher_f')
    adaboost.watcher_f(f_matrix);
end

if isfield(adaboost, 'watcher_c')
    adaboost.watcher_c(C_matrix);
end

for t = 1:adaboost.T
   fprintf('AdaBoostMM Iteration: %d ...\n', t); tic;
   fprintf('\tDoing C\n');
    %% Calculate C_t(i, l) the cost matrix for boosting.
    for i=1:m
        for l=1:k
            yi = adaboost.labels(i);
            if l ~= yi
                C_matrix(i, l) = exp(f_matrix(i, l) - f_matrix(i, yi));
            else
                C_matrix(i, l) = 0;
                for j=1:k,
                    if j == yi
                        continue;
                    end
                    C_matrix(i, l) = C_matrix(i, l) - exp(f_matrix(i, j) - f_matrix(i, yi));
                end
            end
        end
    end
    
    if isfield(adaboost, 'watcher_c')
        adaboost.watcher_c(C_matrix);
    end
    %% Receive the weak classifier
    while true
        fprintf('\tWeak Learning\n');
        H_t = adaboost.weak_learner(adaboost.train_data, C_matrix);
        H_matrix{t} = H_t;
        %% Compute edge \delta_t
        predictions = zeros(m, 1);

        fprintf('\tDoing prediction ...');
        parfor i=1:m,
            predictions(i) = adaboost.weak_predictor(H_t, adaboost.train_data, i); 
        end
        fprintf('done\n');

        fprintf('\tCalculating edge ... ');
        numinator = 0; denominator = 0;

        for i=1:m
            numinator = numinator - C_matrix(i, predictions(i));
            yi = adaboost.labels(i);
            for l=1:k
               if l == yi
                   continue;
               end
               denominator = denominator + exp(f_matrix(i, l) - f_matrix(i, yi));
            end
        end

        delta = numinator/denominator;
        
        if delta > 0
            break;
        end
    end    
    
    
    if isfield(adaboost, 'watcher_delta')
        adaboost.watcher_delta(delta);
    end
    fprintf('%.2f\n', delta);
    %% Compute \alpha_t
    fprintf('\tCalculating alpha ...');
    alpha = 1/2 * log((1+delta)/(1-delta));
    % or ...
    alpha_mat(t) = alpha;
    if isfield(adaboost, 'watcher_alpha')
        adaboost.watcher_alpha(alpha);
    end
    fprintf('%.2f\n', alpha);
    %% Compute next state f_matrix
    fprintf('\tCalculating next state\n');
    for i=1:m,
        target = predictions(i);
        f_matrix(i, target) = f_matrix(i, target) + alpha;
    end
    
    if isfield(adaboost, 'watcher_f')
        adaboost.watcher_f(f_matrix);
    end
    %%
    fprintf('Done %.2fs\n', toc);
end

adaboost_out.alphas = alpha_mat;
adaboost_out.hs     = H_matrix;
adaboost_out.labels = k;
adaboost_out.predict= @(x) predict_adaboostmm(x, alpha_mat, H_matrix, k, adaboost.weak_predictor);

end