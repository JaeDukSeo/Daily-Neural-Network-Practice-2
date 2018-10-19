function [i,j,dire]=contono2(i,j,dirs,CONT);

% CONTONO2 funcion macarronica que indica en que direccion sigue el 
% contorno binario definido en la matriz CONT a partir del punto (ie,je)
% procedente de la direccion k.
%
% OJO: El contorno debe ser continuo, sin bifurcaciones y
%      'totalmente contenido' en la matriz, es decir ideal.
%
% Las direcciones se definen asi:
%
%
%                      1  
%
%                      |  
%
%              4  -- pixel --  2              
%
%                      |   
%
%                      3
%
% El contorno completo quedara codificado mediante el pto inicial y una
% cadena de direcciones.
%
% USO: [i,j,direcc]=contono2(ie,je,dire,CONT);

v=[CONT(i-1,j) CONT(i,j+1) CONT(i+1,j) CONT(i,j-1)]; 

if dirs==1
     v(3)=0;
elseif dirs==2
     v(4)=0;
elseif dirs==3
     v(1)=0;
elseif dirs==4
     v(2)=0;
end

dire=find(v);dire=min(dire);

if dire==1
     i=i-1;j=j;
elseif dire==2
     i=i;j=j+1;
elseif dire==3
     i=i+1;j=j;
elseif dire==4
     i=i;j=j-1;
else
     dire=0;
end     
