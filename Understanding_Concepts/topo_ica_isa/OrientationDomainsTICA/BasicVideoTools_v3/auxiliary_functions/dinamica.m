function [EEEE,aaaa,T]=dinamica(acel,Et,At,t)

% DINAMICA calcula la posicion y la velocidad de un punto en un instante t+dt
% a partir de:
%
%      - La expresion de la aceleracion en funcion de la posicion (x) y el
%        tiempo (parametro p): a(x,p)
%
%      - Las condiciones iniciales (posicion y velocidad en el instante t)
%
%      - Los valores de t y dt.
%
% Integra las ecuaciones mediante Runge-Kutta de orden 5 (Vease TENENBAUM pag.709)
% (La integracion cutre a pelo es catastrofica cuando el campo depende de x) 
%  
%
% USO: [Posicion-velocidad(t+dt),aceleracion(t+dt),traslacion(t)]=dinamica('a(x,p)',[Posicion-velocidad(t+dt)],dt,t);
%
% NOTA: la posicion y la velocidad vienen dadas en un solo vector fila de dimension 6

%t=t+At;
%aaaa=funcion(Et(1:3),t,acel);
%EEEE=[Et(1:3)+Et(4:6)*At+aaaa*At^2 Et(4:6)+aaaa*At];

aaaa=funcion(Et(1:3),t,acel);
%EEEE=[Et(1:3)+Et(4:6)*At Et(4:6)+aaaa*At];

v1=Et(4:6)*At;
w1=funcion(Et(1:3),t,acel)*At;

v2=(Et(4:6)+0.5*w1)*At;
w2=funcion(Et(1:3)+0.5*v1,t+0.5*At,acel)*At;

v3=(Et(4:6)+0.5*w2)*At;
w3=funcion(Et(1:3)+0.5*v2,t+0.5*At,acel)*At;

v4=(Et(4:6)+w3)*At;
w4=funcion(Et(1:3)+v3,t+At,acel)*At;

EEEE=[Et(1:3)+(1/6)*(v1+2*v2+2*v3+v4) Et(4:6)+(1/6)*(w1+2*w2+2*w3+w4)];
T=(1/6)*(v1+2*v2+2*v3+v4);