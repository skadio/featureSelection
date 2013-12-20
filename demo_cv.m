% demo_cv.m
%
% This demo script shows the basic operation of some feature
% selection algorithms studied in J. Pohjalainen, O. Rasanen &
% S. Kadioglu: "Feature Selection Methods and Their Combinations in
% High-Dimensional Classification of Speaker Likability, Intelligibility
% and Personality Traits", Computer Speech and Language, 2014.
% The methods involved in this script are SD, MI and RSFS.
%
% In the demo, a set of artificial features is generated from the Fisher's
% Iris data by first adding noise and then random mapping the data points to
% a higher-dimensional space and replacing some of the features with noise.
% Then the most useful features are sought by different algorithms and
% k-nearest neighbors (KNN) classifier is used to classify the data into
% Iris classes based on the discovered feature subsets.
%
% 5-fold cross validation is performed on the data. For each
% cross-validation partition into training and test subsets, the training
% subset is used both as feature selection data and as training data for
% the eventual system, while the final class labelings are recorded for the
% test subset.
%
% (c) Jouni Pohjalainen & Okko Rasanen
%
% For Mathworks' classification demos on the original Fisher Iris data,
% please see:
%
% http://www.mathworks.se/products/statistics/examples.html?file=
% /products/demos/shipping/stats/classdemo.html
%
% Questions and comments can be sent to jouni.pohjalainen@aalto.fi or
% okko.rasanen@aalto.fi .


load fisheriris

% Convert class label strings into integer labels
specs = unique(species);
labels = zeros(size(species));
for k = 1:length(specs)
    labels(ismember(species,specs(k))) = k;
end

% Add Gaussian noise to the measurement data (original fisheriris is too easy for classification).
noiselevel = 1;
meas = meas+randn(size(meas)).*noiselevel;

% Generate a set of new features through random projection from the
% original 4 Iris features to a d-dimensional feature space.
d = 200;
M = randn(size(meas,2),d);
M = sqrt(ones./(sum((M.*M)')))'*ones(1,size(M,2)).*M; % Normalize M rows
features = meas*M;

% Replace max 50% of the generated features with random noise features
a = 1 + floor(size(features,2)*rand(round(d/2),1));
features(:,a) = randn(size(features,1),length(a));

N = size(features,1);

k = 5;     % k used in KNN classification

% Test KNN classification accuracy with the different feature sets
%using 5-fold randomized cross-validation in training/testing data division
ncv = 5; cvblocksize = N/5;
dataorder = randperm(N);

fprintf('Cross-validated feature selection using SD, MI and RSFS\n');
fprintf('Please see the source code for more information\n');
fprintf('Evaluation started\n');

hypos_orig = zeros(N,1);
hypos_SD = zeros(N,1);
hypos_MI = zeros(N,1);
hypos_RSFS = zeros(N,1);
for cvi=1:ncv
    fprintf('Cross validation partition %d/%d\n',cvi,ncv);
    
    % test indices for this cross validation round
    testidx = dataorder(((cvi-1)*cvblocksize+1):min(N,cvi* ...
        cvblocksize));
    % train indices for this cross validation round
    trainidx = setdiff(1:N,testidx);
    trainidx = trainidx(randperm(length(trainidx)));

    %% Select features using different algorithms
    [F_MI,W_MI] = MI(features(trainidx,:),labels(trainidx),3);
    [F_SD,W_SD] = SD(features(trainidx,:),labels(trainidx),3);
    Ntrain = length(trainidx);
    trainidx1 = trainidx(1:round(Ntrain/2));
    trainidx2 = trainidx((round(Ntrain/2)+1):end);
    [F_RSFS,W_RSFS] = RSFS(features(trainidx1,:),features(trainidx2,:),labels(trainidx1),labels(trainidx2),'verbose',1);

    % perform classification of test data
    hypos_orig(testidx) = KNN(features(trainidx,:),features(testidx,:),labels(trainidx),k);
    hypos_SD(testidx) = KNN(features(trainidx,F_SD(1:10)),features(testidx,F_SD(1:10)),labels(trainidx),k);
    hypos_MI(testidx) = KNN(features(trainidx,F_MI(1:10)),features(testidx,F_MI(1:10)),labels(trainidx),k);
    hypos_RSFS(testidx) = KNN(features(trainidx,F_RSFS),features(testidx,F_RSFS),labels(trainidx),k);
end

fprintf('Original %d features: %0.2f%% correct.\n',size(features,2),sum(hypos_orig == labels)/length(labels)*100);
fprintf('Best 10 features from SD: %0.2f%% correct.\n',sum(hypos_SD == labels)/length(labels)*100);
fprintf('Best 10 features from MI: %0.2f%% correct.\n',sum(hypos_MI == labels)/length(labels)*100);
fprintf('RSFS feature set (%d features): %0.2f%% correct.\n',length(F_RSFS),sum(hypos_RSFS == labels)/length(labels)*100);
