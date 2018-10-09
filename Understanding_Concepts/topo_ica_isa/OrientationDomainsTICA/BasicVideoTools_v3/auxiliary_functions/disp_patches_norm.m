function array=disp_patches_norm(A,m,buf,norm,valor)

% DISP_PATCHES displays the image basis functions at the columns of matrix A
%
% array=disp_patches_norm(A,m,buf,norm,value)
%
%  Usage:
%   A = basis function matrix
%   m = number of rows (of the image blocks)
%   buf = space between patches
%   norm = 0 no normalization is applied 
%        = 1 each image is normalized according to the extreme (max or min) of the input images 
%   value = value given to the space between images
%

[L M]=size(A);

sz=sqrt(L);

% buf=1;

if ~exist('m','var'),
    if floor(sqrt(M))^2 ~= M
        m=floor(sqrt(M/2));
        n=M/m;
    else
        m=sqrt(M);
        n=m;
    end
else
    n=M/m;
end

array = valor*ones(buf+m*(sz+buf),buf+n*(sz+buf));

k=1;

for i=1:m
  for j=1:n
    if norm==1
    clim=max(abs(A(:,k)));
    array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=...
	reshape(A(:,k),sz,sz)/clim;
    else
    array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=...
	reshape(A(:,k),sz,sz);
    end
    k=k+1;
  end
end

% imagesc(array,'EraseMode','none',[-1 1]);
% axis image off
% drawnow
