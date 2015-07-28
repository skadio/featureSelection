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
% test subset. Note that the training data is further divided into two
% halves in order to compute wrapper algorithm performance criterion on
% a set of samples distinct from the classifier training samples during the 
% feature selection stage.
%
% Please see demo_simple.m for a simpler (but unrealistic) demo without 
% the cross validation. 
%
% (c) Jouni Pohjalainen & Okko Rasanen
%
% For Mathworks' classification demos on the original Fisher Iris data,
% please see:
%
% http://www.mathworks.se/products/statistics/examples.html?file=
% /products/demos/shipping/stats/classdemo.html
%
% Questions and comments can be sent to jpohjala@acoustics.hut.fi or
% okko.rasanen@aalto.fi.

load fisheriris meas species 

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
% original 4 features to d dimensions.
d = 200;
M = randn(size(meas,2),d);
M = sqrt(ones./(sum((M.*M)')))'*ones(1,size(M,2)).*M; % Normalize M rows
features = meas*M;

% Replace max 50% of the generated features with random noise features
a = 1 + floor(size(features,2)*rand(round(d/2),1));
features(:,a) = randn(size(features,1),length(a));

fprintf('Feature selection using SD, MI, RSFS, SFS and SFFS\n');
fprintf('Please see the source code for more information\n');
fprintf('Evaluation started\n');

N = size(features,1);

k = 5;     % k used in KNN classification

% Test KNN classification accuracy with the different feature sets
%using 5-fold randomized cross-validation in training/testing data division
ncv = 5; 
cvblocksize = N/5;
dataorder = randperm(N);

hypos_orig = zeros(N,1);
hypos_SD = zeros(N,1);
hypos_MI = zeros(N,1);
hypos_RSFS = zeros(N,1);
hypos_SFFS = zeros(N,1);
hypos_SFS = zeros(N,1);

nfeat_RSFS = 0;
nfeat_SFS = 0;
nfeat_SFFS = 0;

for cvi=1:ncv
    fprintf('Cross validation partition %d/%d\n',cvi,ncv);
    
    % test indices for this cross validation round
    testidx = dataorder(((cvi-1)*cvblocksize+1):min(N,cvi*cvblocksize));
    % train indices for this cross validation round
    trainidx = setdiff(1:N,testidx);
    trainidx = trainidx(randperm(length(trainidx)));
    
    Ntrain = length(trainidx);
    % Divide training data into two halves ("train + dev") 
    trainidx1 = trainidx(1:round(Ntrain/2));
    trainidx2 = trainidx((round(Ntrain/2)+1):end);

    %% Select features using different algorithms
    [F_MI,W_MI] = MI(features(trainidx,:),labels(trainidx),3);
    [F_SD,W_SD] = SD(features(trainidx,:),labels(trainidx),3);
        
    [F_RSFS,W_RSFS] = RSFS(features(trainidx1,:),features(trainidx2,:),labels(trainidx1),labels(trainidx2),'verbose',1);
        
    k_sfs = 5:5:20; % Values of KNN k parameter over which Sequential Forward Selection (SFS) is performed
    t_sfs = 3;      % How many iterations is SFS run beyond the first detected performance maximum? 
    [F_SFS,W_SFS] = SFS(features(trainidx1,:),features(trainidx2,:),labels(trainidx1),labels(trainidx2),k_sfs,t_sfs);
    
    k_sffs = 5:5:20; % Values of KNN k parameter over which Sequential Floating Forward Selection (SFS) is performed
    t_sffs = 3;      % How many iterations is SFFS run beyond the first detected performance maximum? 
    [F_SFFS,W_SFFS] = SFFS(features(trainidx1,:),features(trainidx2,:),labels(trainidx1),labels(trainidx2),k_sffs,t_sffs);

    % perform classification of test data
    hypos_orig(testidx) = KNN(features(trainidx,:),features(testidx,:),labels(trainidx),k);
    hypos_SD(testidx) = KNN(features(trainidx,F_SD(1:10)),features(testidx,F_SD(1:10)),labels(trainidx),k);
    hypos_MI(testidx) = KNN(features(trainidx,F_MI(1:10)),features(testidx,F_MI(1:10)),labels(trainidx),k);
    hypos_RSFS(testidx) = KNN(features(trainidx,F_RSFS),features(testidx,F_RSFS),labels(trainidx),k);
    hypos_SFS(testidx) = KNN(features(trainidx,F_SFS),features(testidx,F_SFS),labels(trainidx),k);
    hypos_SFFS(testidx) = KNN(features(trainidx,F_SFFS),features(testidx,F_SFFS),labels(trainidx),k);
    
    % to compute the average number of features selected by RSFS, SFS and SFFS
    nfeat_RSFS = nfeat_RSFS + length(F_RSFS);
    nfeat_SFS = nfeat_SFS + length(F_SFS);
    nfeat_SFFS = nfeat_SFFS + length(F_SFFS);
end
nfeat_RSFS = nfeat_RSFS/ncv;
nfeat_SFS  = nfeat_SFS/ncv;
nfeat_SFFS = nfeat_SFFS/ncv;

% Print results over all 5 folds
fprintf('Original %d features: %0.2f%% correct.\n',size(features,2),sum(hypos_orig == labels)/length(labels)*100);
fprintf('Best 10 features from SD: %0.2f%% correct.\n',sum(hypos_SD == labels)/length(labels)*100);
fprintf('Best 10 features from MI: %0.2f%% correct.\n',sum(hypos_MI == labels)/length(labels)*100);
fprintf('RSFS feature sets (%0.1f features on average): %0.2f%% correct.\n',nfeat_RSFS,sum(hypos_RSFS == labels)/length(labels)*100);
fprintf('SFS feature sets (%0.1f features on average): %0.2f%% correct.\n',nfeat_SFS,sum(hypos_SFS == labels)/length(labels)*100);
fprintf('SFFS feature sets (%0.1f features on average): %0.2f%% correct.\n',nfeat_SFFS,sum(hypos_SFFS == labels)/length(labels)*100);
