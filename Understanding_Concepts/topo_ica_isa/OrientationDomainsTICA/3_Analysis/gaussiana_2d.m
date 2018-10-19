function G = gaussiana_2d(xxxx,m,a,b,alfa)

% G = gaussiana_2d([x(:) y(:)],m,a,b,alfa)


% Cr = R*C*Rt


R = [cos(alfa) -sin(alfa);sin(alfa) cos(alfa)];
xr = R*[xxxx(:,1)'-m(1);xxxx(:,2)'-m(2)];


G = exp( -((xr(1,:).^2)/a^2 + (xr(2,:).^2)/b^2) )';
G = G/sum(G);