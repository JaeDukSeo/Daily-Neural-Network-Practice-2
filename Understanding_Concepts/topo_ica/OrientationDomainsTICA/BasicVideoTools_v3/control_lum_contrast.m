function im2 = control_lum_contrast(im1,L,C)

% CONTROL_LUM_CONTRAST sets the average luminance and RMSE sinus-like contrast for a natural image  
% 
%  im2 = control_lum_contrast(im1,L,C)
%

im2 = im1;

L1 = mean(im1(:));
A1 = std(im1(:))*sqrt(2);

im2 = im2 - L1;
im2 = im2/A1;

im2 = L + C*L*im2;
