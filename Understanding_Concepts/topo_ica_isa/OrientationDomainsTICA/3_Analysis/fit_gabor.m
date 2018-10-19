function [m_opt,a_opt,b_opt,alfa_opt,f_opt,phase_opt,e] = fit_gabor(B,xx,yy,m0,f0,a0,b0,alfa0,pinta)

% [m_opt,a_opt,b_opt,alfa_opt,f_opt,phase_opt,error] = fit_gabor(B,xx,yy),m0,f0,a0,b0,alfa0,pinta);


D = size(B);
B = B/sum(sum(abs(B)));
x = [xx(:) yy(:)];

gabor = @(p,x) gabor_2d_fit(x,p(1:2),p(3),p(4),p(5),p(6:7),p(8));

p0 = [m0,a0,b0,alfa0,f0];

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

m_opt = p_opt(1:2);
a_opt = p_opt(3);
b_opt = p_opt(4);
alfa_opt = p_opt(5);
f_opt = p_opt(6:7);
phase_opt = p_opt(8);

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