function [S,W] = SFFS(X1,X2,y1,y2,k,t,N)
% function [S,W] = SFFS(X1,X2,y1,y2,k,t,N)
%
% Sequential Floating Forward Selection (wrapper feature selection for classification)
%
% Reference:
% P. Pudil, J. Novovicova, J. Kittler:
% "Floating search methods in feature selection",
% Pattern Recognition Letters, vol. 15, pp. 1119-1125,
% November 1994.
%
% This function performs sequential floating forward selection using two
% data sets, X1 and X2, whose columns correspond to the same features, and
% their associated class labels y1 and y2. Starting from an empty feature
% set, one feature at a time is added based on which feature gives the best
% classification performance when used together with the previously
% selected features.
%
% In this implementation, a k-nearest-neighbor classifier is
% trained with a subset of columns of X1 and evaluated on the same subset
% of columns of X2, and vice versa, possibly for multiple values of the k
% (number of neighbors) parameter. The averaged classification score from
% each test is used to decide whether to include the feature in the
% selected feature set. A tolerance parameter t specifies how many features
% are allowed to be added without improving upon the best classification
% score obtained during the search. The maximum acceptable size of feature
% set can be specified by parameter N. Increasing the size of the feature
% set stops when either of the conditions (related to t and N) is reached.
%
%       Required inputs:
%           X1, X2:                 Two data matrices with the same number
%                                   of columns, equivalent to the total
%                                   number of features. As a special case,
%                                   X1 == X2 (training and testing on the
%                                   same data).
%
%           y1, y2:                 True class labels of the rows of X1 and
%                                   X2
%
%       Optional inputs:
%
%           k:                      Number of neighbors for kNN
%                                   classification. If k is a vector, kNN
%                                   is run on each value in the vector and
%                                   the results are averaged.
%                                   Default value = 5.
%
%           t:                      The number of features that is allowed
%                                   to be included in the selected feature
%                                   set without improving the best overall
%                                   classification score previously seen
%                                   with smaller feature sets.
%                                   Default value = 0.
%
%           N:                      The maximum number of features to be
%                                   selected. Default value equals the
%                                   number of columns (features) in X1 and
%                                   X2.
%
%       Outputs:
%
%           S:                  Indices of the chosen features in the order
%                               of selection
%
%           W:                  Classification score for each nested
%                               feature set
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function implements sequential floating forward selection as a
% "wrapper" feature selection method, that is, it uses a classification
% algorithm to evaluate each different feature subset. For further
% discussion on different types of feature selection algorithms, see for
% example
% J. Pohjalainen, O. Rasanen & S. Kadioglu: "Feature Selection Methods and
% Their Combinations in High-Dimensional Classification of Speaker Likability,
% Intelligibility and Personality Traits", Computer Speech and Language,
% 2015.
%
% NOTE: The algorithm uses unweighted average recall (UAR) as the default
% performance criterion for relevance update. Modify the function calls at
% rows 132-133 and 183-184 in order to introduce your own criterion function.
%
% NOTE 2: The algorithm uses KNN classifier by default. The classifier can
% be changed on rows 136-137 and 189-190.
%
% NOTE 3: The KNN implementation KNNI used here is able to incrementally
% update the distance matrix, which can speed up computation of nested
% feature sets like forward selection uses.
%
% Authors: Jouni Pohjalainen and Okko Rasanen, 2014.
% Mail: jpohjala@acoustics.hut.fi, okko.rasanen@aalto.fi
%
% The algorithm can be freely used for research purposes. Please email your
% bug reports or other comments to one of the above addresses.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<5
    k = 5;
end

if nargin<6
    t = 0;
end

if nargin<7
    N = size(X1,2);
end

S = [];
W = [];
loop = 1+t;
totalbestval = 0;

% KNNI SPECIFIC: initialize KNN distance matrix
D12 = 0;
D21 = 0;

while loop>0

    % EVALUATE THE UNION OF THE CURRENTLY SELECTED FEATURES WITH EACH YET
    % UNSELECTED FEATURE
    
    bestval(length(S)+1) = 0;
    bestidx = 0;
    for i1=1:size(X1,2)
        if ~ismember(i1,S)

            % add the effect of the new feature to existing squared Euclidean distance matrix
            [y1_knn,newD21] = KNNI(X2(:,i1),X1(:,i1),y2,k,1,D21); % Do KNN classification with X2 as training data and X1 as test data
            [y2_knn,newD12] = KNNI(X1(:,i1),X2(:,i1),y1,k,1,D12); % Do KNN classification with X1 as training data and X2 as test data

            averagescore = 0;
            for i2=1:size(y1_knn,2)
                score_1 = uac(y1_knn(:,i2), y1);    % Get classification performance on X1
                score_2 = uac(y2_knn(:,i2), y2);    % Get classification performance on X2
                % Compute overall performance score
                averagescore = averagescore + size(X1,1)*score_1;
                averagescore = averagescore + size(X2,1)*score_2; % Weight according to X2 data size
            end
            averagescore = averagescore/(length(k)*(size(X1,1)+size(X2,1))); % average over two datasets and each different k value
            
            % Check whether obtained score is better than the old one, if
            % yes, set the currently included feature as the best candidate
            % and update the best score
            if averagescore>bestval(length(S)+1)
                bestval(length(S)+1) = averagescore;
                bestidx = i1;

                % KNNI SPECIFIC: store KNN distance matrix that contains the current next best feature
                bestD12 = newD12;
                bestD21 = newD21;

                statstr = ['SFFS(' num2str(length(S)) '): ' num2str(i1) ': ' num2str(averagescore)];
                if averagescore>=totalbestval
                    statstr = [statstr ' *'];
                end
                disp(statstr);
            end
        end
    end

    % INCLUDE THE BEST FEATURE IN THE SET OF SELECTED FEATURES
    
    S = [S bestidx];
    W = [W bestval(length(S))];

    % KNNI SPECIFIC: update KNN distance matrix with the chosen feature
    D12 = bestD12;
    D21 = bestD21;
 
    % "FLOATING" EXCLUSION
    
    % Exclude thus far selected features one at a time and find the least
    % useful one. Repeat this as long as the evaluation score stays better than with
    % the previous best score using a feature set of the same size
    first_exclusion = true;
    while length(S)>2
        exclusion_score = 0; excidx = 0;
        for i1=1:length(S)

            S_x = setdiff(S,S(i1));
            [y1_knn,newD21] = KNNI(X2(:,S_x),X1(:,S_x),y2,k,1); % Do KNN classification with X2 as training data and X1 as test data
            [y2_knn,newD12] = KNNI(X1(:,S_x),X2(:,S_x),y1,k,1); % Do KNN classification with X1 as training data and X2 as test data

            averagescore = 0;
            for i2=1:size(y1_knn,2)
                score_1 = uac(y1_knn(:,i2), y1);    % Get classification performance on X1
                score_2 = uac(y2_knn(:,i2), y2);    % Get classification performance on X2
                % Compute overall performance score
                averagescore = averagescore + size(X1,1)*score_1;
                averagescore = averagescore + size(X2,1)*score_2; % Weight according to X2 data size
            end
            averagescore = averagescore/(length(k)*(size(X1,1)+size(X2,1))); % average over two datasets and each different k value
            
            
            % look for the best feature to exclude
            if averagescore>exclusion_score
                exclusion_score = averagescore;
                excidx = S(i1);
                bestD12 = newD12;
                bestD21 = newD21;
            end
        end

        if (first_exclusion && excidx==bestidx) || (exclusion_score<=bestval(length(S)-1))
            % if the least useful feature is the most recently added
            % feature, or no gain can be obtained by exclusion, don't
            % exclude anything and continue with forward selection
            break;
        else
            % if exclusion leads to better score than what has been previously
            % obtained with a feature set of the same size, exclude the
            % least useful feature
            idx = find(S==excidx);
            S(idx) = [];
            W(idx) = [];

            % KNNI SPECIFIC: update the "current" distance matrix according
            % to the feature that is excluded
            D12 = bestD12;
            D21 = bestD21;

            bestval(length(S)) = exclusion_score;
            loop = 1+t; % reset counter after feature exclusion
            
            statstr = ['Exclusion (' num2str(length(S)) '-1): ' num2str(excidx) ': ' num2str(exclusion_score)];
            if exclusion_score>=totalbestval
                statstr = [statstr ' *'];
            end
            disp(statstr);
        end
        first_exclusion = false;
    end
    
    % STOPPING CONDITIONS

    % quit if specified maximum number of features has been achieved
    if length(S)>=N
        break;
    end

    if bestval(length(S))<=totalbestval
        loop = loop - 1; % the feature set after inclusion and exclusion did not improve upon the overall best score => decrease the counter
    else
        loop = 1+t; % improvement was obtained by the new feature set => reset the counter to its starting value
        totalbestval = bestval(length(S));
    end    

end

% BACK OFF TO THE BEST PERFORMING FEATURE SET

S = S(1:end-(t+1)+loop);
W = W(1:end-(t+1)+loop);
