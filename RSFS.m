function [S,W,AW] = RSFS(features_train,features_dev,labels_train,labels_dev,varargin)
% function [S,W,AW] = RSFS(features_train,features_dev,labels_train,labels_dev,varargin)
%
% Performs Random Subset Feature Selection (RSFS) for the given set of
% features divided into training (features_train) and development
% (features_devel) sets.
%
%       Required inputs:
%           features_train:     N X F matrix of features used to train KNN
%                                   classifier in RSFS, where N is the number
%                                   of samples and F is the total number of original
%                                   features.
%           features_devel:     M x F matrix of features used to test KNN
%                                   classifier in RSFS, where M is the
%                                   number of samples in the development
%                                   set.
%           labels_train:       N X 1 vector of class labels for train set samples.
%           labels_devel:       M X 1 vector of class labels for dev set samples.
%
%       Optional inputs:
%
%           'max_iters'         Maximum number of iterations that RSFS is
%                                   run (default: 300 000).
%
%           'max_delta'         Stopping criterion for feature set size
%                                   fluctuation (std(Siz)/mean(Siz) where Siz
%                                   is the size of the feature set during last
%                                   1000 iterations. Default: 0.005 (i.e., 0.5%
%                                   fluctuation in relative size). Set this to
%                                   a negative value if you want to run RSFS up to
%                                   max_iters.
%           'n_dummyfeats'      Number of dummy features used to define chance
%                                   level relevance statistics (default: 100)
%           'k'                 Number of neighbors used in KNN
%                                   classification (default: 3)
%           'verbose'           Print intermediate results on command line? (0/1)
%
%       Outputs:
%
%           S:                  Indices of the chosen features (from range [1-F])
%           W:                  Relevance values of the chosen features.
%           AW:                 Relevance values of all input features
%
% Example 1: Stop at 20000 iterations or when the feature set has 1% fluctuation in size 
%
%   [S,W] = RSFS(features_train,features_devel,labels_train,labels_devel,'max_iters',20000,'max_delta',0.01)
%
% Example 2: Run all 300000 iterations (the default value) despite possible
% earlier convergence in feature set size
%
%   [S,W] = RSFS(features_train,features_devel,labels_train,labels_devel,'max_delta',-Inf)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A very brief description of the algorithm:
%   RSFS repeatedly performs k-nearest neighbors (KNN) classification using
%   a randomly chosen subset of all possible features. Relevance of each
%   feature is updated according to its participation into well-performing
%   random subsets. Ultimately, the "good" features are chosen from the
%   entire feature pool by comparing the relevance values of the features
%   to random walk statistics. The algorithm is executed until 1) maximum
%   allowed fluctuation in feature set size is reached (max_delta) or 2)
%   the maximum number of iterations is reached (max_iters).
%
% Please see J. Pohjalainen, O. Rasanen & S. Kadioglu: "Feature Selection Methods and
% Their Combinations in High-Dimensional Classification of Speaker Likability,
% Intelligibility and Personality Traits", Computer Speech and Language, 2015,
%  or
% Räsänen O. & Pohjalainen J., "Random subset feature selection in automatic recognition
% of developmental disorders, affective states, and level of conflict from speech",
% Proc. Interspeech'2013, Lyon, France, 2013, for more details.
%
% Above papers should be also cited in case of RSFS is being used in a
% publication.
%
% NOTE: The algorithm uses unweighted average recall (UAR) as the default
% performance criterion for relevance update. Modify the existing code at rows
% 160 and 161 in order to introduce your own criterion function.
%
% NOTE 2: The algorithm uses KNN classifier by default. The classifier can
% be changed on row 140.
%
% Author: Okko Rasanen, 2013. Mail: okko.rasanen@aalto.fi
%
% The algorithm can be freely used for research purposes. Please send your
% bug reports or other comments to okko.rasanen@aalto.fi.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

max_iters=300000; n_dummyfeats=100; max_delta=0.005; k_neighbors=3; verbose=0;
argidx = 1;
while argidx<length(varargin) && ischar(varargin{argidx})
  switch varargin{argidx}
   case 'max_iters'
       max_iters = varargin{argidx+1};
   case 'n_dummyfeats'
       n_dummyfeats = varargin{argidx+1};
   case 'max_delta'
       max_delta = varargin{argidx+1};
   case 'k'
       k_neighbors = varargin{argidx+1};
   case 'verbose'
       verbose = varargin{argidx+1};
  end
  argidx = argidx + 2;
end

number_of_features = size(features_train,2);

%% Perform mean and std normalization to the features
for k = 1:number_of_features
    features_train(:,k) = features_train(:,k)-mean(features_train(:,k));
    features_train(:,k) =  features_train(:,k)./std(features_train(:,k));
    if all(isnan(features_train(:,k)))
        features_train(:,k) = 1;
    end
end

for k = 1:number_of_features
    features_dev(:,k) = features_dev(:,k)-mean(features_dev(:,k));
    features_dev(:,k) =  features_dev(:,k)./std(features_dev(:,k));
    if all(isnan(features_dev(:,k)))
        features_dev(:,k) = 1;
    end
end


% Define the number of classes in the training data
N_classes = max(labels_train);

% Format vectors for the relevance values of true features and dummy features
relevance = zeros(number_of_features,1);
dummy_relevance = zeros(n_dummyfeats,1);

% Calculate the number of features and dummy features used per each RSFS
% iteration as the square root of the entire feature pool size.
feats_to_take = round(sqrt(number_of_features));
dummy_feats_to_take = round(sqrt(n_dummyfeats));

% Number of features chosen, stored separately for each iteration
feat_N = zeros(max_iters,1);

% Class-specific classification accuracies across the entire execution 
% (needed for the expected value of the UAR performance criterion).
totcorrect = zeros(N_classes,1);
totwrong = zeros(N_classes,1);

% Run RSFS "max_iters" times or up to the stabilization of the feature size
iteration = 1;
deltaval = Inf;
while(iteration <= max_iters && deltaval > max_delta)
    
    %feature_indices = randi(number_of_features,1,feats_to_take);  % Select a subset of features randomly
    feature_indices = 1 + floor(number_of_features*rand(1,feats_to_take));  % Select a subset of features randomly
    
    % ----> REPLACE YOUR OWN CLASSIFIER BELOW <----
    class_hypos = KNN(features_train(:,feature_indices),features_dev(:,feature_indices),labels_train,k_neighbors);
    
    % Measure class-specific accuracies
    correct = zeros(N_classes,1);
    wrong = zeros(N_classes,1);
    
    for j = 1:length(labels_dev)
        if(labels_dev(j) == class_hypos(j))
            correct(labels_dev(j)) = correct(labels_dev(j))+1;
        else
            wrong(labels_dev(j)) = wrong(labels_dev(j))+1;
        end
    end
    
    % Update cumulative class-specific accuracies since first iteration
    totcorrect = totcorrect+correct;
    totwrong = totwrong+wrong;
    
    % ----> REPLACE YOUR OWN PERFORMANCE CRITERION BELOW <----
    % Measure Unweighted Average Recall (UAR) of the current iteration
    performance_criterion = mean(correct./(correct+wrong).*100); 
    expected_criterion_value = mean(totcorrect./(totcorrect+totwrong).*100); 
   
    % Update relevance values for true features according to 
    % r' <- r+c-E(c) where r is current relevance value
    % c is the current function value and E(c) is the expectation of the
    % criterion function value.
    relevance(feature_indices) = relevance(feature_indices)+(performance_criterion-expected_criterion_value);
    
    % Select a random dummy feature subset
    %dummy_indices = randi(n_dummyfeats,dummy_feats_to_take,1);
    dummy_indices = 1 + floor(n_dummyfeats*rand(dummy_feats_to_take,1));
    
    % Update dummy feature relevancies similarly to the true features
    dummy_relevance(dummy_indices) = dummy_relevance(dummy_indices)+(performance_criterion-expected_criterion_value);
    
    % Get the probability that a true feature has a higher relevance
    % than the dummy features
    probs = normcdf(relevance,mean(dummy_relevance),std(dummy_relevance));
    
    % Find how many features exceed p > 0.99 (i.e., are "chosen" so far).
    feat_N(iteration) = length(find(probs > 0.99));
    
    if(mod(iteration,1000) == 0)
        if(verbose == 1)
            deltaval = std(feat_N(iteration-999:iteration))/mean(feat_N(iteration-999:iteration));
            fprintf('RSFS: %d features chosen so far (iteration: %d/%d). Delta: %0.6f.\n',feat_N(iteration),iteration,max_iters,deltaval);
        end
    end
    iteration = iteration+1;
end

% Find features that are better than random with p < 0.01
S = find(probs > 0.99);
% Get corresponding weights
W = relevance(S);  

AW = relevance;
% Sort features to the order of relevance
[W,o] = sort(W,'descend');
S = S(o);
