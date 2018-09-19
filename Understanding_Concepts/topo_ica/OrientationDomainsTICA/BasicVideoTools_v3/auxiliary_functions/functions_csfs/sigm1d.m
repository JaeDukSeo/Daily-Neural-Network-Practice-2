function y=sigm1d(x,x0,s);

% SIGM1D da el valor de una sigmoide 1D:
%      
%                         1
%     s(x,xo,s) = -------------------       
%                  1 + exp((x-xo)/s)
%
% en el conjunto de puntos x que se le den
%
% USO: y=sigm1d(x,xo,s);

y=1./(1+exp((x-x0)/s));