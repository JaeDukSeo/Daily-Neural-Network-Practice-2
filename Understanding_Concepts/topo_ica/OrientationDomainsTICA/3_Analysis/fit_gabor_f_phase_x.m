function [m_opt,a_opt,b_opt,alfa_opt,f_opt,phase_opt,e] = fit_gabor_f_phase_x(B,xx,yy,m0,f0,a0,b0,alfa0,pinta)

% [m_opt,a_opt,b_opt,alfa_opt,f_opt,phase_opt,error] = fit_gabor(B,xx,yy),m0,f0,a0,b0,alfa0,pinta);


D = size(B);
B = B/sum(sum(abs(B)));
x = [xx(:) yy(:)];

gabor = @(p,x) gabor_2d_fit(x,m0,a0,b0,alfa0,p(1:2),p(3));

p0 = [f0];

% initialization of phase

e1 = sum(abs(B(:) - gabor([p0 0],x)));
e2 = sum(abs(B(:) - gabor([p0 pi/2],x)));

if e2 > e1
   p0 = [p0 0];  
else
   p0 = [p0 pi/2]; 
end
    
% Optimization

p_opt = lsqcurvefit(gabor,p0,x,B(:));

m_opt = m0;
a_opt = a0;
b_opt = b0;
alfa_opt = alfa0;
f_opt = p_opt(1:2);
phase_opt = p_opt(3);

BBB= B;
e = sum(  abs( BBB(:) - gabor(p_opt,x) )  )/sum(B(:)>0)/max(B(:));
% % e = sum(  abs( abs(BBB(:)) - gaussiana_2d([x(:) y(:)],m_opt,a_opt/2,b_opt/2,alfa_opt) )  )/sum(BBB(:)>0)/max(BBB(:))
if pinta ==1
figure,mesh(BBB),axis square,axis([0 D(1) 0 D(2) -0.3*max(BBB(:)) max(BBB(:))])
figure,mesh(reshape(gabor(p_opt,x),D(1),D(2))),axis square,axis([0 D(1) 0 D(2) -0.3*max(BBB(:)) max(BBB(:))])
figure,mesh(BBB-reshape(gabor(p_opt,x),D(1),D(2))),axis square,axis([0 D(1) 0 D(2) -0.3*max(BBB(:)) max(BBB(:))])
end

function G = gabor_2d_fit(xxxx,m,a,b,alfa,f,phase);

% G = gabor_2d_fit([x(:) y(:)],m,a,b,alfa,f,phase);

% Cr = R*C*Rt

R = [cos(alfa) -sin(alfa);sin(alfa) cos(alfa)];
xr = R*[xxxx(:,1)'-m(1);xxxx(:,2)'-m(2)];

s = sin(2*pi*( f(1)*(xxxx(:,1)'-m(1)) + f(2)*(xxxx(:,2)'-m(2))) + phase );
G = exp( -((xr(1,:).^2)/a^2 + (xr(2,:).^2)/b^2) )'.*s';
G = G/sum(abs(G));