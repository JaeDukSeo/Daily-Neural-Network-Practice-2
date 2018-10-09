function im = norm_image(im);

% norm_image 
% 

mea = mean(im(:));
st = std(im(:));
im = (im - (mea - st))/(2*st);
