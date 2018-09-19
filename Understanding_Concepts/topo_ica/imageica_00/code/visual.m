function visual( A, mag, cols )
% visual - display a basis for image patches
%
% A        the basis, with patches as column vectors
% mag      magnification factor
% cols     number of columns (x-dimension of map)
%

% Get maximum absolute value (it represents white or black; zero is gray)
maxi=max(max(abs(A)));
mini=-maxi;

% This is the side of the window
dim = sqrt(size(A,1));

% Helpful quantities
dimm = dim-1;
dimp = dim+1;
rows = size(A,2)/cols;
if rows-floor(rows)~=0, error('Fractional number of rows!'); end

% Initialization of the image
I = maxi*ones(dim*rows+rows-1,dim*cols+cols-1); 

for i=0:rows-1
  for j=0:cols-1
    
    % This sets the patch
    I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim) = ...
			reshape(A(:,i*cols+j+1),[dim dim]);
  end
end

I = imresize(I,mag);

figure;
colormap(gray(256));
iptsetpref('ImshowBorder','tight'); 
subplot('position',[0,0,1,1]);
imshow(I,[mini maxi]);
truesize;  

