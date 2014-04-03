function [I,C] = l2i(L)
% function [I,C] = l2i(L)
% Convert (class) label array L into an array of integer-valued labels I.
% Also return the list of original labels C which can be indexed by I.
%
% Example:
% load fisheriris
% gscatter(meas(:,1),meas(:,2),species);
% disp('Press enter');
% pause;
% [Is,Cs] = l2i(species);
% gscatter(meas(:,1),meas(:,2),Is);
% [f,w]=MI(meas,Is);

% J.P. 030414 

C = unique(L);
I = zeros(size(L));
for k = 1:length(C)
    I(ismember(L,C(k))) = k;
end
