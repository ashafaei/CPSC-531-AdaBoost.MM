%% Configuration

if ~exist('script_config', 'var')
    script_config = struct();
end

script_config.train_path = 'cifar-10-batches-mat/data_batch_1.mat';
script_config.test_path  = 'cifar-10-batches-mat/test_batch.mat';

script_config.T          = 200;

%% Loading
if ~isfield(script_config, 'train_data') || ~isfield(script_config, 'test_data'),
    fprintf('Loading data... ');
    script_config.train_data = load(script_config.train_path);
    script_config.test_data  = load(script_config.test_path);
    
    script_config.train_data.images = convert_to_imgs(script_config.train_data.data);
    script_config.test_data.images  = convert_to_imgs(script_config.test_data.data);
    
    script_config.train_data.labels = script_config.train_data.labels + 1;
    script_config.test_data.labels  = script_config.test_data.labels  + 1;
    
    script_config.train_data.gray_images = zeros([size(script_config.train_data.images, 1)...
        size(script_config.train_data.images, 2) size(script_config.train_data.images, 4)], 'uint8');
    
    script_config.test_data.gray_images  = zeros([size(script_config.test_data.images, 1)...
        size(script_config.test_data.images, 2) size(script_config.test_data.images, 4)], 'uint8');
    
    for i=1:size(script_config.train_data.images, 4),
       script_config.train_data.gray_images(:, :, i) = rgb2gray(script_config.train_data.images(:, :, :, i)); 
    end
    
    for i=1:size(script_config.test_data.images, 4),
       script_config.test_data.gray_images(:, :, i) = rgb2gray(script_config.test_data.images(:, :, :, i)); 
    end
    
    gf = fspecial('gaussian',[1 1], 1);
    
    script_config.train_data.blurred = imfilter(script_config.train_data.gray_images, gf, 'replicate');
    script_config.test_data.blurred  = imfilter( script_config.test_data.gray_images,  gf, 'replicate');
 
    clear gf;
    
    fprintf('done.\n');
end

%%

adaboost = struct();

adaboost.train_data     = script_config.train_data;
adaboost.weak_learner   = @ (train_data, C_matrix) wl_hog(train_data, C_matrix);
adaboost.weak_predictor = @wp_hog;

% adaboost.watcher_f      = @(x) fprintf('F data feed\n');

adaboost.n_classes      = 10;
adaboost.n_examples     = size(script_config.train_data.data, 1);
adaboost.T              = script_config.T;
adaboost.labels         = script_config.train_data.labels;

tic;
adaboost_out = learn_adaboostmm(adaboost);
fprintf('AdaBoost T:%d took %.2fs\n', adaboost.T, toc);

%% Test the output

confusion_mat = zeros(adaboost.n_classes, adaboost.n_classes);

predictions = zeros(size(script_config.test_data.data, 1));

parfor i=1:size(script_config.test_data.data, 1),
    predictions(i) = adaboost_out.predict(script_config.test_data.gray_images(:, :, i));
end

for i=1:size(script_config.test_data.data, 1),
    target_class = script_config.test_data.labels(i);
    confusion_mat(target_class, predictions(i)) = confusion_mat(target_class, predictions(i)) + 1;
end

precision = sum(diag(confusion_mat))/sum(sum(confusion_mat));

fprintf('The average test precision: %.3f\n', precision);

%% Test the output on training

confusion_mat_train = zeros(adaboost.n_classes, adaboost.n_classes);
predictions_test = zeros(size(script_config.test_data.data, 1), 1);

parfor i=1:size(script_config.test_data.data, 1),
    predictions_test(i) = adaboost_out.predict(script_config.train_data.gray_images(:, :, i));
end

for i=1:size(script_config.test_data.data, 1),
    target_class = script_config.train_data.labels(i);
    confusion_mat_train(target_class, predictions_test(i)) = ...
        confusion_mat_train(target_class, predictions_test(i)) + 1;
end


precision = sum(diag(confusion_mat_train))/sum(sum(confusion_mat_train));

fprintf('The average training precision: %.3f\n', precision);
