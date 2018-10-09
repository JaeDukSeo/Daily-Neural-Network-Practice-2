function v=sinusoidal_flow(pos,v0,f1,fase);

% SINUSOIDAL_FLOW computes the speed at a location in a sinusoidal flow field
% with spatial frequency "f" and amplitude "v"
%
%   v(1)=v0(1)*sin(2*pi*(f(1)*x+f(2)*y)+phase);
%   v(2)=v0(2)*sin(2*pi*(f(1)*x+f(2)*y)+phase);
%
% This function is intended to be used with dots_sequence.m and a proper
% initialization (see example_random_dots_sequence.m)
%
% USE:  v=sinusoidal_flow([x y],v0,f,fase);

v=[v0(1)*sin(2*pi*(f1(1)*pos(1)+f1(2)*pos(2))+fase) v0(2)*sin(2*pi*(f1(1)*pos(1)+f1(2)*pos(2))+fase)];

