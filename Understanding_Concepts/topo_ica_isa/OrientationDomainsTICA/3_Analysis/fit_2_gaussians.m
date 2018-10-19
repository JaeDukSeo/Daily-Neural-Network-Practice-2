function [m_opt,a_opt,b_opt,alfa_opt,e] = fit_2_gaussians(B,xx,yy,pinta);

% [m_opt,a_opt,b_opt,alfa_opt,error] = fit_2_gaussians(B,fx,fy,pinta);


D = size(B);
B = abs(B)/sum(sum(abs(B)));

% initialization of mean

    BB = imresize(B,round([D(1)/1.5 D(2)/1.5]));
    BB = imresize(BB,D,'bicubic');
    
    p = find(BB==max(BB(:)));
    
    xxx = xx(p(1));
    yyy = yy(p(1));
    
    m0 = [xxx yyy];
    
% Initialization of sigmas
    
    a0 = sqrt(    length(find(BB > 0.15*max(BB(:))))*(xx(1,2)-xx(1,1))*(yy(2,1)-yy(1,1))/2   )/2;
    b0 = a0;
    cond = BB > 0.15*max(BB(:));
    
% Initialization of angle 

    alfa0 = 0;
    
% Optimization

x = [xx(:) yy(:)];
guasianas = @(p,x) two_gaussianas_2d(x,[p(1) p(2)],p(3),p(4),p(5));

BBB = BB.*cond;
BBB = BBB/sum(BBB(:));
p_opt = lsqcurvefit(guasianas,[m0 a0 b0 alfa0],x,BBB(:));

m_opt = p_opt(1:2);
a_opt = p_opt(3);
b_opt = p_opt(4);
alfa_opt = p_opt(5);

e = sum(  abs( abs(BBB(:)) - guasianas(p_opt,x) )  )/sum(BBB(:)>0)/max(BBB(:));
% % e = sum(  abs( abs(BBB(:)) - gaussiana_2d([x(:) y(:)],m_opt,a_opt/2,b_opt/2,alfa_opt) )  )/sum(BBB(:)>0)/max(BBB(:))
if pinta ==1
figure,mesh(BBB),axis square,axis([0 D(1) 0 D(2) -0.3*max(BBB(:)) max(BBB(:))])
figure,mesh(reshape(guasianas(p_opt,x),D(1),D(2))),axis square,axis([0 D(1) 0 D(2) -0.3*max(BBB(:)) max(BBB(:))])
figure,mesh(BBB-reshape(guasianas(p_opt,x),D(1),D(2))),axis square,axis([0 D(1) 0 D(2) -0.3*max(BBB(:)) max(BBB(:))])
end

function G = two_gaussianas_2d(xxxx,m,a,b,alfa)

% G = two_gaussiana_2d(xxxx,m,a,b,alfa)

% Cr = R*C*Rt

R = [cos(alfa) -sin(alfa);sin(alfa) cos(alfa)];
xr = R*[xxxx(:,1)'-m(1);xxxx(:,2)'-m(2)];
xr2 = R*[xxxx(:,1)' + m(1); xxxx(:,2)' + m(2)];


G1 = exp( -((xr(1,:).^2)/a^2 + (xr(2,:).^2)/b^2) )';
G2 = exp( -((xr2(1,:).^2)/a^2 + (xr2(2,:).^2)/b^2) )';
G = (G1+G2)/sum(G1+G2);
