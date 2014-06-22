function [ua,ac] = uac(labels,truelabels)
% [ua,ac] = uac(labels,truelabels)
% Given a predicted labeling and a true reference labeling,
% computes the unweighted average recall (class-average
% classification accuracy) and the overall accuracy.
% ua - unweighted average recall
% ac - classification accuracy
% labels - predicted labels
% truelabels - actual class labels

% JP 2013

label_index = unique(truelabels);
nlabels = length(label_index);

% evaluate classification performance
Ncorrect = zeros(nlabels,1);  % the number of correctly classified instances for each class
Ninstances = ones(nlabels,1); % the number of instances for each class (actual class)
for i2=1:nlabels
    idx = ismember(truelabels,label_index(i2)); % indices of the instances of this class
    if any(idx) % if there are any instances
        Ncorrect(i2) = sum(ismember(labels(idx),label_index(i2))); % count the correctly classified instances
        Ninstances(i2) = sum(idx);
    end
end
ac = (sum(Ncorrect)/sum(Ninstances)) * 100;
ua = mean(Ncorrect./Ninstances) * 100;
