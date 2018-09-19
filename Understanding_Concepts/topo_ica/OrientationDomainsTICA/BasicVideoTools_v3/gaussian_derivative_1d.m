function [g,x] = gaussian_derivative_1d(fs,N,x0,sigma,order)

% GAUSSIAN_DERIVATIVE_1D computes the n-th derivative of a normalized Gaussian
% located at x0 with width sigma.
%
% Location and width are given in physical units (e.g. degrees or seconds)
% according to the meaning of the selected sampling frequency.
% 
% The derivative is given by the corresponding Hermite polynomial located
% at that point multiplied by the Gaussian.
% This function uses: 
%     hermite.m from Avan Suinesiaputra
%     spatio_temp_freq_domain.m from Jesus Malo 
%
% [g,x] = gaussian_derivative_1d(N,fs,x0,sigma,n)
%
% Example: 
%      Biphasic impulse response located at t0 = 1 sec with width sigma = 100 msec
%
%      fs = 30;    % Sampling frequency in Hz
%      N = 500;    %   -> total time = N/fs = 16 secs 
%      x0 = 1;     % Location in time
%      sigma = 0.1 % Width
%      n = 1;      % First derivative -> Biphasic
%      [g,x] = gaussian_derivative_1d(fs,N,x0,sigma,n);
% 
%

[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(N,N,1,fs,fs,1);
x = x(1,:);

    c = sigma * sqrt(2);
    gx = exp(-((x-x0).^2)/(2*sigma^2)) ./ (sqrt(2*pi)*sigma);
    % gx(gx<eps*max(x)) = 0;
    Hn = hermite(order, (x-x0) ./ c);
    g = Hn .* gx .* (-1/c).^order;
