function [X, whiteningMatrix, dewhiteningMatrix, m] = ...
				whiten_hyva( x, rdim );
            
% whiten_hyva - computes whitening transform from samples
%
% INPUT variables:
% x                  samples
% rdim               reduced dimensionality
%
% OUTPUT variables:
% xw                 the whitened data (in the rotated domain) as column vectors
% whiteningMatrix    transformation of patch-space to xw-space
% dewhiteningMatrix  inverse transformation
% m                  mean vector
%
% [xw, whiteningMatrix, dewhiteningMatrix, m] = whiten_hyva( x, rdim );


%----------------------------------------------------------------------
% Subtract local mean gray-scale value from each patch
%----------------------------------------------------------------------

fprintf('Subtracting local mean...\n');

% % A la mia
% m = mean(x')';
% x = x - repmat(m,1,size(x,2));

% A la Hyva
x = x - ones(size(x,1),1)*mean(x);
m = mean(mean(x));


%----------------------------------------------------------------------
% Reduce the dimension and whiten at the same time!
%----------------------------------------------------------------------

% Calculate the eigenvalues and eigenvectors of covariance matrix.
fprintf ('Calculating covariance...\n');
covarianceMatrix = x*x'/size(x,2);
[E, D] = eig(covarianceMatrix);
%figure,semilogy(diag(D))

% Sort the eigenvalues and select subset, and whiten
fprintf('Reducing dimensionality and whitening...\n');
[dummy,order] = sort(diag(-D));

E = E(:,order(1:rdim));
d = diag(D); 
d = real(d.^(-0.5));
D = diag(d(order(1:rdim)));
X = D*E'*x;

whiteningMatrix = D*E';
dewhiteningMatrix = E*D^(-1);

return;