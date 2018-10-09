function G=sens_gabor3d_freq(tam_fx,tam_fy,tam_ft,fsx,fsy,fst,fxo,fyo,fto,delta_fx,delta_fy,delta_ft)

% SENS_GABOR3D_FREQ computes the squared frequency response of 3D Gabor sensors 
% tuned to certain spatio-temporal frequency with certain frequency widths.
% The Gaussian window in the Fourier domain is aligned with the frequency axes.
%
% The expression for the frequency response is:
%
%   g(f) = B*exp(-(f-f0)'*A^-1*(f-f0))
%
% where A is diagonal with the widths in the directions fx, fy, y ft
%
%           / Delta_fx^2      0          0       \
%       A = |                                    |
%           |    0       Delta_fy^2      0       |
%           |                                    |
%           \    0            0      Delta_ft^2  /
%
% The B parameter is a normalization factor to have sum(g) = 1 
%
%      B=1/(4*pi^(3/2)*Delta_fx*Delta_fy*Delta_ft)
%
% The routine requires the sizes of the discrete domain (rows,columns,frames)
%
% SYNTAX: G = sens_gabor3d_freq(columns,rows,frames,fsx,fsy,fst,fxo,fyo,fto,delta_fx,delta_fy,delta_ft)
%

[fx,fy]=freqspace([tam_fx tam_fy],'meshgrid');
fx=fx*fsx/2;
fy=fy*fsy/2;
[ft1,ft2]=freqspace(tam_ft);
ft=ft1*fst/2;

gs1=exp(-(fx-fxo).^2/delta_fx^2-(fy-fyo).^2/delta_fy^2);
gs2=exp(-(fx+fxo).^2/delta_fx^2-(fy+fyo).^2/delta_fy^2);

G=zeros(tam_fy,tam_fx*tam_ft);

for i=1:tam_ft
    g1=gs1.*(exp(-(ft(i)-fto).^2/delta_ft^2));
    g2=gs2.*(exp(-(ft(i)+fto).^2/delta_ft^2));    
    G=metefot(G,g1+g2,i,1);
    %i
end

B=1/(4*pi^(3/2)*delta_fx*delta_fy*delta_ft);
G=B*G;

