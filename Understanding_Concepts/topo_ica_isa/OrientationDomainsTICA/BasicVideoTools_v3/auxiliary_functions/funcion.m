function y=funcion(x,p,funci)

% FUNCION evalua la funcion definida en el dominio X y dependiente de
% los parametros P, en el punto Xo con los parametros Po y la forma
% funcional dada en 'f(x,p)'.
%
% El caracter escalar, vectorial o matricial del resultado Y depen-
% dera del caracter de Xo y Po, y de las operaciones definidas en 
% 'f(x,p)'
%
% USO: y=funcion(xo,po,'f(x,p)'); 


eval(['y=',funci,';'])