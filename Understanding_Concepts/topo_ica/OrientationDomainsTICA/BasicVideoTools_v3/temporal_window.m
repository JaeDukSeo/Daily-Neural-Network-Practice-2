function vent_temp=ventana_temp(t,delta_t)

% 
% TEMPORAL_WINDOW is convenient to reduce edge effects when filtering spatio-temporal movies.
% It is intended to be applied to the movie (plain .*) in the spatio-temporal domain before filtering 
% in the Fourier domain. 
% The returned window linearly increases during a certain delta_t temporal period (in sec.) 
% and decreases similarly at the final delta_t seconds.
% 
% Requires the 3D temporal domain "t" computed by "spatio_temp_freq_domain.m"
%
% temp_w = temporal_window(t,delta_t);
% 

inc = delta_t;
vent_temp = double((t<=inc).*t/inc)+double((t>inc)&(t<=(maxi(t)-inc)))-double((t>(maxi(t)-inc)).*(( t-(maxi(t)) )/inc));