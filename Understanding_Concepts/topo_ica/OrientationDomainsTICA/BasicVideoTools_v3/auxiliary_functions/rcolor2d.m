function [r,fil,rf]=rcolor2D(fs,N,f1,f2,o,a,t,d);

%'RCOLOR2D' genera ruido bidimensional con una cierta anchura de banda
% (frecuencial y de orientacion).
% Esto se consigue partiendo de alguna clase de ruido blanco (a elegir 
% entre ruido uniforme, gaussiano, poissoniano o definido por el usuario)
% y aplicando un filtrado pasa banda al ruido blanco para conseguir el
% ruido coloreado.
% 
% USO: [r,fil,rf]=rcolor2d(fs,N,fmin,fmax,orientac,anchura_angular,tipo,[parametros del ruido blanco inicial]);
% 
% Tipos de ruidos blancos de partida:
%
%       1.......Ruido uniforme (introducir la desviacion como parametro)
%
%       2.......Ruido gaussiano (introducir la desviacion como parametro)
%
%       3.......Ruido poissoniano (introducir la desviacion como parametro)
%
%       4.......Ruido con funcion densidad introducida manualmente
%               (En este caso, el parametro a introducir es el rango donde 
%                se extiende la parte significativa de la funcion densidad 
%                de probabilidad: [xm xM] )
%
% NOTA 1: Logicamente las restricciones de banda se cargan las caracteris-
%         ticas estadisticas del ruido blanco de partida (?).  
%         Para asegurarnos de que es lo que hemos calculado, lo mejor es
%         analizar a posteriori el ruido mediante 'FDISDENS'.       
%
% NOTA 2: Para el ruido uniforme y el ruido gaussiano no se introduce el
%         parametro 'media' porque en general el filtrado pasa banda anula
%         la componente de continua -> media=0, asi que directamente se 
%         parte de ruidos con media cero.     


if t==1
   r=runif(0,d,N,N);  
elseif t==2
   r=rnormal(0,d,N,N); 
elseif t==3
   if d<9.5
      r=rpoisson(d^2,50,N,N);
   else
      r=rnormal(0,d,N,N);
   end
else  
   r=rmanual(d(1),d(2),50,1,N,N); 
end

fil=pasaband(fs,N,f1,f2,o,a);
rf=filtra(r,ifft2(fil),1);
fil=fftshift(fil);