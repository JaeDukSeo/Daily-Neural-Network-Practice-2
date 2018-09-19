% ----------------------------------------------------
% EXAMPLE OF NEWTONIAN SEQUENCES 
% (a way to know almost everything about the sequence)
%
% See additional demos in BasicVideoTools:
%   demo_motion_programs.m 
%   example_rendom_dots_sequence.m
% See additional motion estimation toolbox:
%   http://isp.uv.es/Video_coding.html
% ----------------------------------------------------
%
% Classical mechanics and lambertian optics allows us to generate 
% sequences in which the vertices of a rigid solid have known 2D (retinal) 
% speeds (optical flow) and known (actual) 3D speeds. 
% All this tunable information is very useful as ground truth to check the
% result of motion estimation algorithms and stablish their limits. 
% 
% Newtonian dynamics of the rigid solid allows to define actual 3D motion
% by integrating the motion equations given the force fields and the initial
% conditions.
% This moving solid can be illuminated from certain direction and projected
% onto some camera filming the scene from some point of view. 
% 
% BasicVideoTools comes with routines to generate such sequences:
%
%  newtonian_sequence      - Sequence from a rigid solid moving in force fields
%     elipso3              - Definition of facets and trapezoids of an ellipsoid
%     dinam_tr             - Translation and rotation newtonian dynamics (Runge-Kutta integration) 
%     pintael2             - Illumination of facets and projection onto the camera
% 
% In this script we generate different 10 movies from the same moving object 
% from 5 differents points of view and illuminated from 5 different directions.
%
% Object motion has the following properties/constraints:
% -------------------------------------------------------
%
% - Initially the object has a certain position and speed (linear and angular).
%   
% - The object is subject to an acceleration field (linear and angular acceleration)
%
% - In this example 3D motion may be confined in a 3D bounding box so that
%   the object bounces (with certain restitution coeffcient) when it hits the walls. 
%
%   NOTE: the integration of the motion equations is done using a temporal 
%   frequency M times faster than the acquisition frame rate (in this case M=5),
%   and we use a 4th order Runge-Kutta algorithm.
%
% (Given a fixed 3D motion) the acquired sequence is determined by:
% -----------------------------------------------------------------
%
% - Point of view of the scene (azimut and elevation)
%
% - The object is illuminated by a distant source in some illumination direction
%   (with certain -different- azimut and elevation). 
%
% - The object is lambertian.
%
% - We take N frames with temporal sampling frequency ft (in Hz)
%
% - Image plane has certain limits
%
% - Image plane is sampled using Nx pixels/dimension
%
% In this script we play with all these parameters. Have fun!
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECT MOTION: initial conditions and acceleration fields
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Posicion y velocidad
    
         pos_vel=[6 6 6 -6 -1.5 0];
    
    % Angulo y velocidad angular
    
         ang_vel_ang=[0 0 0 0 0.5 0];

% Aceleraciones (campos de fuerza). Dos casos concretos (si no comentas el segundo, es el que se aplicara):

    % Atraccion por un punto
       fact_atrac=150;
       punto=[3 3 3];
       aceleracion_lineal=['-',num2str(fact_atrac),'*[x(1)-',num2str(punto(1)),' x(2)-',num2str(punto(2)),' x(3)-',num2str(punto(3)),']/(sqrt(sum([x(1)-',num2str(punto(1)),' x(2)-',num2str(punto(2)),' x(3)-',num2str(punto(3)),'].^2)))^3']
       aceleracion_angular='0.1*[1 1 0]'; 
        % Paredes de confinamiento (limites xm xM, ym yM, zm zM) y coeficiente de
        % restitucion (perdida de energia en los rebotes)
            limites=3*[-1 7 -1 7 -1 7];
            coef_restit=0.7;

       
    % Gravedad (en el eje z)
       gravedad=9.8;
       aceleracion_lineal='-[0 0 9.8]';
       aceleracion_angular='[0 0 0]'; 

       % Paredes de confinamiento (limites xm xM, ym yM, zm zM) y coeficiente de
       % restitucion (perdida de energia en los rebotes)
           limites=[-1 7 -1 7 -1 7];
           coef_restit=0.7;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SEQUENCE PARAMETERS: illumination, view point, sampling frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
% Puntos de vista (azimut y elevacion)

vistas=[0 30;15 30;30 30;45 30;45+15 30];

% Iluminacion (azimut y elevacion)

ilumin=[15 60];

% Frecuencia de muestreo temporal (Intervalo temporal entre fotogramas) y numero de fotogramas

fst=20; % 20 fotogramas por segundo
dt=1/fst; 
N=125;

% Limites del plano imagen y numero de pixels

limit_imagen=1.8*[-3 8 -3 8];
Nx=100;

% Figura donde se representa

fig=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOP OVER THE DIFFERENT SEQUENCE CONDITIONS: newtonian_sequence.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:5

[S,Peli,Posic_veloc,acel_CM,T,Angul_velocang,acelang,R,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,...
       se_ve,p2dx,p2dy]=newtonian_sequence(aceleracion_lineal,aceleracion_angular,pos_vel,ang_vel_ang,...
       0,dt,N,5,limites,1,coef_restit,vistas(i,:),ilumin,1,limit_imagen,3,3,3,0,0,0,4,Nx,0.95,fig);

       eval(['peli',num2str(i),'=S;']);
       
        %% You can also save frame-by-frame data 
        %
        % [S,Peli,Posic_veloc,acel_CM,T,Angul_velocang,acelang,R,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,...
        %        se_ve,p2dx,p2dy]=newtonian_sequence(aceleracion_lineal,aceleracion_angular,pos_vel,ang_vel_ang,...
        %        0,dt,N,5,limites,1,coef_restit,vistas(i,:),ilumin,1,limit_imagen,1.5,1.5,1.5,0,0,0,4,Nx,0.95,fig,...
        %        'c:\disco_portable\',['lala',num2str(i)]);
        %    
        % [PEL,ESTT,ACET,TTTT,ESTR,ACER,RRR,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,se_ve,p2dx,p2dy]=...
        %     cargapel(Nx,Nx,['c:\disco_portable\lala',num2str(i)],[1 N]);  
        %    
        %     eval(['peli',num2str(i),'=S;']);
   
end

for i=6:10

[S,Peli,Posic_veloc,acel_CM,T,Angul_velocang,acelang,R,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,...
       se_ve,p2dx,p2dy]=newtonian_sequence(aceleracion_lineal,aceleracion_angular,pos_vel,ang_vel_ang,...
       0,dt,N,5,limites,1,coef_restit,vistas(1,:),vistas(i-5,:),1,limit_imagen,3,3,3,0,0,0,4,Nx,0.95,fig);
       eval(['peli',num2str(i),'=S;']);
end
clear S Peli
close all

for i=1:10
    eval(['mov_3_',num2str(i),' = then2now(peli',num2str(i),',Nx);'])
    eval(['clear peli',num2str(i),';'])
end

pelis = zeros(2*Nx+2*4,5*Nx+(5*4),N);
for i=1:N
    fotos = 64*ones(Nx+4,Nx+4,10);
    for j=1:10
        eval(['fotos(3:end-2,3:end-2,j) = mov_3_',num2str(j),'(:,:,i);'])
    end
    fot = [fotos(:,:,1) fotos(:,:,2) fotos(:,:,3) fotos(:,:,4) fotos(:,:,5);
           fotos(:,:,6) fotos(:,:,7) fotos(:,:,8) fotos(:,:,9) fotos(:,:,10)];
    pelis(:,:,i) = fot;   
end
clear mov_*
M = build_achrom_movie(pelis,0,64,5*Nx+5*4,1);