function umb=umbinc3(c,cu,k,m,alf,sig)
  
% UMBINC3 calcula el umbral incremental para los contrastes que se le pasen
% segun la expresion:
%
%     umb=cu+(1-sigm1d(c,alfa*cu,sigma)).*(k*c.^m-cu)
%
%     -A*exp(-abs(log(c/(alf*cu)))/sig)
% 
% umb=umbinc3(c,cu,k,m,alfa,sigma)

%umb=cu+(c./((alf*cu)+c)).*(k*c.^m-cu);
%umb=cu+(1-sigm1d(c,alf*cu,sig)).*(k*c.^m-cu);
%umb=cu+(1-1./(1+exp(log10(c/(alf*cu))/sig))).*(k*c.^m-cu)-A*cu*exp(-abs(log(c/(alf*cu)))/sig);

umb=(cu-k*cu^m)*(1./(1+exp(log10(c/(alf*cu))/sig)))+(k*c.^m).*(1-1./(1+exp(log10(c/(0.9*cu))/(sig/2))));