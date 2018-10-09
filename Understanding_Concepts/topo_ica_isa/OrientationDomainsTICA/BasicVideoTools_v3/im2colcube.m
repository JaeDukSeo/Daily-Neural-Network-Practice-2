function I = im2colcube(A,dim,n)

% 
% I = im2colcube(A,dim,opt)
%
% A = cube (sequence or hyperspectral image) of M rows, N cols and K frames (or spectral bands)
%
% dim = [m n], where m = number of rows in a block, and n = number of cols in a block
%
% opt = 1 or 2, to select between the following options (see im2col):
%       if opt =1 => spatially distinct patches
%       if opt =2 => spatially sliding patches
%


[a b c]=size(A);

if n==1
    
        
    iaux = im2col(A(:,:,1),[dim(1),dim(2)],'distinct');

    I=zeros(dim(1)*dim(2)*(c-1),size(iaux,2));
    I=[iaux;I];

    for i=2:c
        
        iaux=im2col(A(:,:,i),[dim(1),dim(2)],'distinct');
        I((i-1)*prod(dim)+1:i*prod(dim),:) = iaux;
        
    end
    
    
else
    
    iaux=im2col(A(:,:,1),[dim(1),dim(2)],'sliding');

    I=zeros(dim(1)*dim(2)*(c-1),size(iaux,2));
    I=[iaux;I];

    for i=2:c
        
        iaux=im2col(A(:,:,i),[dim(1),dim(2)],'sliding');
        I((i-1)*prod(dim)+1:i*prod(dim),:) = iaux;
        
    end

end