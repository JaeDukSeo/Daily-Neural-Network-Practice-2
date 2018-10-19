function [L,C] = lumin_contrast(im)

% LUMIN_CONTRAST computes the average luminance and sinusoid-like contrast for natural images
% 
%       L = mean(image)
%       C = std(image)*sqrt(2)/L
%
% [L,C] = lumin_contrast(im);
%

       L = mean(im(:));
       C = std(im(:))*sqrt(2)/L;