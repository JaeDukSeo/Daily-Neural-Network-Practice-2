function [YR,pm,pM,pR] = positive_luminance(Y,R)

% POSITIVE_LUMINANCE clips the off-range values in the vector of luminances Y
% 
% [Y_range,p_min,p_max,p_range] = positive_luminance(Y,range)
%
% 

pm = double(Y<R(1));
pM = double(Y>R(2));

pR = double((Y>R(1)) & (Y<R(2)));

YR = R(1)*pm + Y.*pR + R(2)*pM;