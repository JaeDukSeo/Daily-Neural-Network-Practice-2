function v=radial_flow(pos,pto_central,v_min,v_max)

% RADIAL_FLOW computes a radial speed with regard to a focus of expansion 
% The speed linearly increases with the distance (in deg) to the focus of expansion. 
% The proportionality constant is v_max (in deg/sec).
%
% This function is intended to be used with dots_sequence.m and a proper
% initialization (see example_random_dots_sequence.m)
%
% USE: v = radial_flow(location,location_foe,v_min,v_max);
%
%  location     = [x y]     (location of the considered point)
%  location_foe = [x_0 y_0] (location of the focus of expansion)
%  v = v_min + v_max*|location-location_foe|
%
%

dist=sqrt(sum((pos-pto_central).^2));
mod_v=v_max*dist+v_min;
dir_v=pos-pto_central;

%v=mod_v*dir_v/dist;
v=mod_v*dir_v;