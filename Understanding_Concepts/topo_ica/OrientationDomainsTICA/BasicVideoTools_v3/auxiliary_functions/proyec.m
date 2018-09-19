function [x2d,A]=proyec(viu,angulo_sub,pto_fijac,x3d);

% PROYEC calcula la proyeccion 2D de unos puntos 3D desde un punto de vista particular
% mediante una transformacion proyectiva que puede contener perspectiva.
%
% Los N puntos 3D se introducen como una matriz de N*3 elementos. 
% Los N puntos 2D se introducen como una matriz de N*2 elementos. 
% Por defecto introducir:
%
%        - angul_sub = 0      
%        - pto_fijac = [0 0 0] (aparece en el punto [0 0] del dominio 2D)
% 
% El resultado de la aplicacion de la matriz de proyeccion se escala mediante los
% factores de la ultima componente!.
%
% USO: [x2d,M_proyeccion(4*4)]=proyec([azimut elevacion],angulo_subtendido(perspect),pto_fijacion,x3d); 


x=x3d(:,1);
y=x3d(:,2);
z=x3d(:,3);
A=viewmtx(viu(1),viu(2),angulo_sub,pto_fijac);
x4d=[x y z ones(length(z),1)]';
x2d=A*x4d;
x2=x2d(1,:)';
y2=x2d(2,:)';
w=x2d(4,:)';
x2d=[x2./w y2./w];
