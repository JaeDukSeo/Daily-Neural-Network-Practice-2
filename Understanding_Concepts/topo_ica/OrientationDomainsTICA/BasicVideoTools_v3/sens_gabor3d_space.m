function [Gs,Gc]=sens_gabor3d_space(Nx,Ny,Nt,fsx,fsy,fst,xo,yo,to,phase,fxo,fyo,fto,delta_x,delta_y,alfa,delta_t)

% SENS_GABOR3D_SPACE computes sine and cosine Gabor receptive fields in the spatio-temporal domain.
% The corresponding Gaussian windows are aligned with the spatiotemporal axes.
% The expression for the sine-Gabor is:
%
%   G_s(p) = B*exp(-(p-p0)'*A^-1*(p-p0)).*sin(2*pi*(f*(p-p0))+phase)
%
% where p - p0 = ( R(alpha)*(x-x0,y-y0), t-t0), and A is diagonal with the widths in the (rotated) directions x, y, and t
%
%           / Delta_x^2      0          0      \
%       A = |                                  |
%           |    0       Delta_y^2      0      |
%           |                                  |
%           \    0            0      Delta_t^2 /
%
% The B parameter is a normalization factor to have sum(G_s.^2) = 1 
% The cosine-Gabor is the same using cosine instead of sine.
%
% The routine requires the sizes of the discrete domain (rows,columns,frames) 
% and the corresponding sampling frequencies (fy,fx,ft)
%
% SYNTAX: [Gs,Gc] = sens_gabor3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo,yo,to,phase,fxo,fyo,fto,delta_x,delta_y,alpha,delta_t)
%

[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(Ny,Nx,Nt,fsx,fsy,fst);

R = [cos(alfa) -sin(alfa);sin(alfa) cos(alfa)];
xr = R*[x(:)'- xo; y(:)'- yo];
xid = [x(:)'- xo; y(:)'- yo];
[fil,col] = size(x);

xxr = reshape(xr(1,:)',fil,col);
yyr = reshape(xr(2,:)',fil,col);
xxid = reshape(xid(1,:)',fil,col);
yyid = reshape(xid(2,:)',fil,col);
exp2d=exp(-(xxr).^2/delta_x^2-(yyr).^2/delta_y^2-(t-to).^2/delta_t^2);

Gs = exp2d.*sin(2*pi*(fxo*(xxid)+fyo*(yyid)+fto*(t-to))+phase);
Gc = exp2d.*cos(2*pi*(fxo*(xxid)+fyo*(yyid)+fto*(t-to))+phase);
% Gc = exp(-(x-xo).^2/delta_x^2-(y-yo).^2/delta_y^2-(t-to).^2/delta_t^2).*cos(2*pi*(fxo*(x-xo)+fyo*(y-yo)+fto*(t-to))+phase);

Gs = Gs/sqrt(sum(Gs(:).^2));
Gc = Gc/sqrt(sum(Gc(:).^2));