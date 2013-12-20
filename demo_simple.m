% demo_simple.m
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
% Note that in this simplified demo, the same set of data samples is used  
% to perform feature selection, to train the KNN classifier and to evaluate it. 
% For a more realistic classification scenario, please see the 
% demo_cv.m where division to separate training, development and testing data is
% accomplished by using a cross-validated evaluation scheme.
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
% original 4 features to d dimensions.
d = 200;
M = randn(size(meas,2),d);
M = sqrt(ones./(sum((M.*M)')))'*ones(1,size(M,2)).*M; % Normalize M rows
features = meas*M;

% Replace max 50% of the generated features with random noise features
a = 1 + floor(size(features,2)*rand(round(d/2),1));
%a = randi(size(features,2),round(d/2),1);
features(:,a) = randn(size(features,1),length(a));

k = 5;     % k used in KNN classification

fprintf('Feature selection using SD, MI and RSFS\n');
fprintf('Please see the source code for more information\n');
fprintf('Evaluation started\n');

%% Select features using different algorithms
[F_MI,W_MI] = MI(features,labels,3);
[F_SD,W_SD] = SD(features,labels,3);

[F_RSFS,W_RSFS] = RSFS(features,features,labels,labels,'verbose',1);

%% Test KNN classification accuracy with the different feature sets using the same data points for training and testing
% (note that k = 1 always leads to 100% accuracy without an independent test set). 

hypos_orig = KNN(features,features,labels,k);
fprintf('Original %d features: %0.2f%% correct.\n',size(features,2),sum(hypos_orig == labels)/length(labels)*100);
hypos_SD = KNN(features(:,F_SD(1:10)),features(:,F_SD(1:10)),labels,k);
fprintf('Best 10 features from SD: %0.2f%% correct.\n',sum(hypos_SD == labels)/length(labels)*100);
hypos_MI = KNN(features(:,F_MI(1:10)),features(:,F_MI(1:10)),labels,k);
fprintf('Best 10 features from MI: %0.2f%% correct.\n',sum(hypos_MI == labels)/length(labels)*100);
hypos_RSFS = KNN(features(:,F_RSFS),features(:,F_RSFS),labels,k);
fprintf('RSFS feature set (%d features): %0.2f%% correct.\n',length(F_RSFS),sum(hypos_RSFS == labels)/length(labels)*100);
