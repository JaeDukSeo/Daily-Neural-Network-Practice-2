function [X, whiteningMatrix, dewhiteningMatrix] = ...
				data( samples, winsize, rdim );
% data - gathers patches from the gray-scale images
%
% INPUT variables:
% samples            total number of patches to take
% winsize            patch width in pixels
% rdim               reduced dimensionality
%
% OUTPUT variables:
% X                  the whitened data as column vectors
% whiteningMatrix    transformation of patch-space to X-space
% dewhiteningMatrix  inverse transformation
%

rand('seed', 0);
  
%----------------------------------------------------------------------
% Gather rectangular image windows
%----------------------------------------------------------------------

% We have a total of 13 images.
dataNum = 13;

% This is how many patches to take per image
getsample = floor(samples/dataNum);

% Initialize the matrix to hold the patches
X = zeros(winsize^2,getsample*dataNum);

sampleNum = 1;  
for i=(1:dataNum)

  % Even things out (take enough from last image)
  if i==dataNum, getsample = samples-sampleNum+1; end
  
  % Load the image
  I = imread(['../data/' num2str(i) '.tiff']);

  % Normalize to zero mean and unit variance
  I = double(I);
  I = I-mean(mean(I));
  I = I/sqrt(mean(mean(I.^2)));
  
  % Sample 
  fprintf('Sampling image %d...\n',i);
  sizex = size(I,2); sizey = size(I,1);
  posx = floor(rand(1,getsample)*(sizex-winsize-2))+1;
  posy = floor(rand(1,getsample)*(sizey-winsize-1))+1;
  
  for j=1:getsample
    X(:,sampleNum) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
			posx(1,j):posx(1,j)+winsize-1),[winsize^2 1]);
    sampleNum=sampleNum+1;
  end 
  
end

%----------------------------------------------------------------------
% Subtract local mean gray-scale value from each patch
%----------------------------------------------------------------------

fprintf('Subtracting local mean...\n');
X = X-ones(size(X,1),1)*mean(X);

%----------------------------------------------------------------------
% Reduce the dimension and whiten at the same time!
%----------------------------------------------------------------------

% Calculate the eigenvalues and eigenvectors of covariance matrix.
fprintf ('Calculating covariance...\n');
covarianceMatrix = X*X'/size(X,2);
[E, D] = eig(covarianceMatrix);

% Sort the eigenvalues and select subset, and whiten
fprintf('Reducing dimensionality and whitening...\n');
[dummy,order] = sort(diag(-D));
E = E(:,order(1:rdim));
d = diag(D); 
d = real(d.^(-0.5));
D = diag(d(order(1:rdim)));
X = D*E'*X;

whiteningMatrix = D*E';
dewhiteningMatrix = E*D^(-1);

return;

