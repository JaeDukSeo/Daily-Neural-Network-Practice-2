function I = col2imcube(B,dim1,dim2)

% I = col2imcube(B,dim1,dim2)
% B image in columns
% dim1 = [size of block number_of_spectral_bands]
% dim2 = [number_of_rows number_of_columns] in the global image (for one
% spectral band)




I=zeros(dim2(1),dim2(2),dim1(2));

for i=1:dim1(2)
    
    iaux=col2im(B(1+dim1(1)^2*(i-1):+dim1(1)^2*(i-1)+dim1(1)^2,:),[dim1(1) dim1(1)],[dim2(1),dim2(2)],'distinct');
    
    I(:,:,i) = iaux;
    
end