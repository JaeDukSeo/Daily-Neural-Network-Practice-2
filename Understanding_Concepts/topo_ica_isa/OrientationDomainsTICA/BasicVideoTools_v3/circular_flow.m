function v=circular_flow(pos,pto_central,v_min,v_max)

% CIRCULAR_FLOW computes the speed at some location assuming a flow field
% with direction tangent to circles centered at some "location_of_center"
% and absolute value depending linearly on the radius (in deg).
% The proportionality constant is v_max (in deg/sec).
%
% This function is intended to be used with dots_sequence.m and a proper
% initialization (see example_random_dots_sequence.m)
%
% USE: v=circular_flow(location,location_of_center,v_min,v_max);
%
%  location     = [x y]     (location of the considered point)
%  location_center = [x_0 y_0] (location of the center)
%  v = v_min + v_max*|location-location_center|
%

dist=sqrt(sum((pos-pto_central).^2));
mod_v=v_max*dist+v_min;
p=pos-pto_central;

dir_v=[0 0];

if (p(1)==0)&(p(2)==0)
   dir_v=[0 0];
   v=[0 0];
else
    if p(2)~=0
       dir_v(1)=1/sqrt(1+(p(1)/p(2))^2);
       dir_v(2)=-(p(1)/p(2))*dir_v(1);
    else
       dir_v(2)=1/sqrt(1+(p(2)/p(1))^2);
       dir_v(1)=-(p(2)/p(1))*dir_v(2);
    end

mn=sqrt(sum((dir_v).^2));
v=mod_v*dir_v/mn;

pv=p_vect([p 0],[v 0]);

if pv(3)<0
   v=-v; 
end    

end

