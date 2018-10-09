function v=lateral_flow(pos,y0,v_max)

% LATERAL_FLOW computes horizontal speeds in flows from ego-motion orthogonal to the optical axis
%
% In this case we will assume that the upper part of the scene (up to location "y0") 
% is far away (no motion) and the lower part of the scene is (linearly)
% closer, giving rise to faster speeds (up to v_max).
%
% This function is intended to be used with dots_sequence.m and a proper
% initialization (see example_random_dots_sequence.m)
%
% USE: v = lateral_flow(location,y0,v_max);
%
%  v = v_max*|y-y0|
%
%

dist = max([0 pos(2)-y0]);
mod_v = v_max*dist;
dir_v = [1 0];

v=mod_v*dir_v;