%Simple code for ICA of images
%Aapo Hyv�rinen, for the book Natural Image Statistics

function [W err Jm] = ica_mich(Z,n,maxIter,convergencecriterion)

if ~exist('convergencecriterion','var')
    convergencecriterion = 1e-10;
end

%------------------------------------------------------------
% input parameters settings
%------------------------------------------------------------
%
% Z                     : whitened image patch data
%
% n = 1..windowsize^2-1 : number of independent components to be estimated


%------------------------------------------------------------
% Initialize algorithm
%------------------------------------------------------------

%create random initial value of W, and orthogonalize it
W = orthogonalizerows(randn(n,size(Z,1))); 

%read sample size from data matrix
N=size(Z,2);

%------------------------------------------------------------
% Start algorithm
%------------------------------------------------------------

%writeline('Doing FastICA. Iteration count: ')

iter = 0;
notconverged = 1;

% times = zeros(1,maxIter);

while notconverged & (iter<maxIter) %maximum of 2000 iterations

  tic
  iter=iter+1;
  
  
  % Store old value
  Wold=W;        

  %-------------------------------------------------------------
  % FastICA step
  %-------------------------------------------------------------  

    % Compute estimates of independent components 
    Y = W*Z; 
    
    
    % Jm(iter) = univ_negentropy(Y(1,:),mean(Y(1,:)),std(Y(1,:))) + univ_negentropy(Y(2,:),mean(Y(2,:)),std(Y(2,:)));
    Jm = 0;
    disp(['Iteration ' num2str(iter)])
    toc
    % Use tanh non-linearity
    gY = tanh(Y);
    
    % This is the fixed-point step. 
    % Note that 1-(tanh y)^2 is the derivative of the function tanh y
    W = gY*Z'/N - (mean(1-gY'.^2)'*ones(1,size(W,2))).*W;  

  % (mean(1-gY'.^2)'*ones(1,size(W,2))) approx the identity
    
    % Orthogonalize rows or decorrelate estimated components
    W = orthogonalizerows(W);

  % Check if converged by comparing change in matrix with small number
  % which is scaled with the dimensions of the data
  err(iter) = norm(abs(W*Wold')-eye(n),'fro');
  if err(iter) < convergencecriterion * n; 
        notconverged=0; 
  end
  
  if (iter-floor(iter/100)*100)==0
%       iter
      err(iter)
  end
  
  toc
%   times(iter) = t;
  
end %of fixed-point iterations loop








