function [features,weights] = MI(features,labels,Q)
% function [features,weights] = MI(features,labels,Q)
% Estimates the mutual information between features and associated class labels using a quantized feature space.
%
%   Inputs:
%           features:       N x F sized matrix of features, where N is the number of samples and F is the number of features        
%           labels:         N x 1 sized vector of class labels corresponding to each sample
%           Q:              the number of quantization levels used for the features (default = 12)
%
%   Outputs:
%           features:       F x 1 sized vector of feature indices in the
%                              descending order of relevance.
%           weights:        F x 1 sized vector of feature relevances (MIs) in the
%                              descending order.
%
% Author: Okko Rasanen, 2013. Mail: okko.rasanen@aalto.fi
%
% The algorithm can be freely used for research purposes. 
%
% Please see J. Pohjalainen, O. Rasanen & S. Kadioglu: "Feature Selection Methods and 
% Their Combinations in High-Dimensional Classification of Speaker Likability, 
% Intelligibility and Personality Traits", Computer Speech and Language, 2015, for more details.

if nargin <3
    Q = 12;
end

edges = zeros(size(features,2),Q+1);

% Compute feature-specific quantization bins so that each bin has approximately equal number of
% samples in the training set
for k = 1:size(features,2)
    
    minval = min(features(:,k));
    maxval = max(features(:,k));
    if minval==maxval
        continue;
    end
    
    quantlevels = minval:(maxval-minval)/500:maxval;
    
    N = histc(features(:,k),quantlevels);
    
    totsamples = size(features,1);
    
    N_cum = cumsum(N);
    
    edges(k,1) = -Inf;
    
    stepsize = totsamples/Q;
    
    for j = 1:Q-1
        a = find(N_cum > j.*stepsize,1);
        edges(k,j+1) = quantlevels(a);
    end
    
    edges(k,j+2) = Inf;
end

% Quantize data according to the obtained bins
S = zeros(size(features));
for k = 1:size(S,2)
    S(:,k) = quantize(features(:,k),edges(k,:))+1;   
end

% Compute mutual information (MI) between the quantized features and
% the class labels
I = zeros(size(features,2),1);
for k = 1:size(features,2)   
    I(k) = computeMI(S(:,k),labels,0);
end

% Sort features into descending order

[weights,features] = sort(I,'descend');

%% EOF


function [I,M,SP] = computeMI(seq1,seq2,lag)
% function [I,M,SP] = computeMI(seq1,seq2,lag)
% Computes the mutual information (MI) between seq1 and seq2 at the
% given delay (lag) between the sequences.
%
%   Inputs:
%
%       seq1:   a discrete sequence of length N
%       seq2:   a discrete sequence of length N
%       lag:    the number of elements that seq1 is delayed with respect to
%               seq2 (a positive or negative integer). Default = 0;


if nargin <3
    lag = 0;
end

if(length(seq1) ~= length(seq2))
    error('Input sequences are of different length');
end


% Count the frequency and probability of each symbol in seq1
lambda1 = max(seq1);
symbol_count1 = zeros(lambda1,1);

for k = 1:lambda1
    symbol_count1(k) = sum(seq1 == k);
end

symbol_prob1 = symbol_count1./sum(symbol_count1)+0.000001;


% Count the frequency and probability of each symbol in seq2
lambda2 = max(seq2);
symbol_count2 = zeros(lambda2,1);

for k = 1:lambda2
    symbol_count2(k) = sum(seq2 == k);
end

symbol_prob2 = symbol_count2./sum(symbol_count2)+0.000001;

% Compute the joint occurrence frequencies of symbol pairs at the given lag

M = zeros(lambda1,lambda2);
if(lag > 0)
    for k = 1:length(seq1)-lag
        loc1 = seq1(k);
        
        loc2 = seq2(k+lag);
        
        M(loc1,loc2) = M(loc1,loc2)+1;
    end
else
    for k = abs(lag)+1:length(seq1)
        loc1 = seq1(k);
        
        loc2 = seq2(k+lag);
        
        M(loc1,loc2) = M(loc1,loc2)+1;
    end
end

% Product of individual state probabilities as a matrix
SP = symbol_prob1*symbol_prob2';

% Pair joint probability
M = M./sum(M(:))+0.000001;

% Compute MI
I = sum(sum(M.*log2(M./SP)));

function y = quantize(x, q)
x = x(:);
nx = length(x);
nq = length(q);
y = sum(repmat(x,1,nq)>repmat(q,nx,1),2);
