function [EE,aa,T,EEr,aar,RR]=dinam_tr(ptos0,acel,acela,E,Er,ric,At,t,cond,Dom,cres);

% DINAM_TR translation and rotation dynamics of the rigid solid.
% DINAM_TR computes (1) the position and speed of the center of mass, (2) the angle
% and the angular speed, and the rotation to matrix (at the time t+dt) from:
%
%   - The initial conditions (at time t): position-speed (center of mass), and
%     angle-angular-speed (with regard to the center of mass)
%
%   - The expression of the linear acceleration and the angular acceleration.
%     Acceletations are given as functions of the location (x) and the time parameter (p)
%          'a(x,p)'  = linear acceleration
%          'aa(x,p)' = angular acceleration
%
%   - The values for t (initial time) and dt (time between frames).
%
%   - The spatial locations that define the elipsoid (elipso3.m).
%
%   - The size of the bounding box where the solid is confined (it bunces when hits the walls).
%
%   - Coeffcient of restitution.
%
%
% USE: [Posicion-velocidadT(t+dt),aceleracionT(t+dt),Traslac(t),Angulo-velocidadA(t+dt),aceleracionA(t+dt),Rotacion(t)]=... 
%                                            dinam_rt('a(x,p)',[Posicion-velocidad(t+dt)],dt,t,caja?,dim_caja,coef_restit);
%
% Note: the location and the speed are given in 1*6 row vectors
%

[EE,aa,T]=dinamica(acel,E,At,t);
[EEr,aar,RR]=dinamrot(acela,Er,At,t);

Np=size(ptos0);Np=Np(1);
for i=1:Np
    ptos(i,:)=(RR*ric*(ptos0(i,:))'+EE(1:3)')';
end

ddti=At;

if cond>0
      con=sesalep(EE(1:3),Dom);
      if con>0
          [Ei,Ti,Eri,Ri1,ddti]=rebota2(ptos0,ric,E,Er,acel,acela,ddti,t+At-ddti,Dom,cres);
          [EE,aa,T]=dinamica(acel,Ei,ddti,t+At-ddti);
          [EEr,aar,RR]=dinamrot(acela,Eri,ddti,t+At-ddti);
          for i=1:Np
                ptos(i,:)=(RR*Ri1*ric*(ptos0(i,:))'+EE(1:3)')';
          end
          con=sesalep(EE(1:3),Dom); 
          if con>0
             [Ei,Ti,Eri,Ri2,ddti]=rebota2(ptos0,Ri1*ric,Ei,Eri,acel,acela,ddti,t+At-ddti,Dom,cres);
             [EE,aa,T]=dinamica(acel,Ei,ddti,t+At-ddti);
             [EEr,aar,RR]=dinamrot(acela,Eri,ddti,t+At-ddti);
             for i=1:Np
                  ptos(i,:)=(RR*Ri2*Ri1*ric*(ptos0(i,:))'+EE(1:3)')';
             end
             con=sesalep(EE(1:3),Dom); 
             if con>0
                  [Ei,Ti,Eri,Ri3,ddti]=rebota2(ptos0,Ri2*Ri1*ric,Ei,Eri,acel,acela,ddti,t+At-ddti,Dom,cres);
                  [EE,aa,T]=dinamica(acel,Ei,ddti,t+At-ddti);
                  [EEr,aar,RR]=dinamrot(acela,Eri,ddti,t+At-ddti);
                  for i=1:Np
                     ptos(i,:)=(RR*Ri3*Ri2*Ri1*ric*(ptos0(i,:))'+EE(1:3)')';
                  end
                  RR=RR*Ri3*Ri2*Ri1;
                  EE=Ei;
                  EEr=Eri;                              
             else
                  RR=RR*Ri2*Ri1;
                  EE=Ei;
                  EEr=Eri;
             end
          else
             RR=RR*Ri1;
             EE=Ei;
             EEr=Eri;
          end       
      end
end
T=EE(1:3)-E(1:3);