function [seq,locations,speeds]=dots_sequence(initial_locations,sizes,colors,flow,Nx,Nt,fs,ft);

% DOTS_SEQUENCE generates a sequence from some initial objects and a flow field.
%
% The basic procedure to generate a random-dots sequence is the following: 
%
% (0) Initialization: 
%     Take a number of objects (rectangles of certain size and color) at 
%     random locations.
%
% (1) Generate an image (frame of the sequence) from the objects data
%     (size, color and location).
%
% (2) Compute (or assign) the 2d (retinal) speed of each object.
%
% (3) Update the locations at a future frame according to the (known) speed 
%     of each object: x(t+1) = x(t) + v(x,t)/ft
%     where x(t) and v(x,t) are the location and speed at time t, and ft is
%     the temporal sampling frequency.
%     We assume that size and color of objects are preserved along the sequence. 
%     Go to step (1) and repeat.
%
% DOTS_SEQUENCE uses BasicVideoTools functions to implement the non-trivial steps:
%
% (1) generate_frame.m
%
% (2) Characteristic spatially-variant flow fields where v can be obtained from x:
%
%       radial_flow.m  (as in ego-motion parallel to the optical axis)
%      lateral_flow.m  (as in ego-motion normal to the optical axis)     
%     circular_flow.m  (the headache-psychic flow)
%   sinusoidal_flow.m  (for those that love to analyze everything into Fourier components)
%      
%     For physics-based fields see XXXX
%
% USE:   [seq,locations,speeds]=dots_sequence(initial_locations,sizes,colors,flow,Nx,Nt,fs,ft);
%


seq = generate_frame(initial_locations,sizes,colors,Nx,Nx,fs,0);

xx = initial_locations;

locations(:,:,1) = xx;

N_objects = length(xx(:,1));

for j=2:Nt
   j
   d=[];
   da=[];

   for i=1:N_objects
            
     % despx=(a1(1)*cos(fase1+2*pi*(f1(1)*pos(i,1)+f1(2)*pos(i,2)))+a2(1)*cos(fase2+2*pi*(f2(1)*pos(i,1)+f2(2)*pos(i,2)))+a3(1)*cos(fase3+2*pi*(f3(1)*pos(i,1)+f3(2)*pos(i,2))))/ft;
     % despy=(a1(2)*cos(fase1+2*pi*(f1(1)*pos(i,1)+f1(2)*pos(i,2)))+a2(2)*cos(fase2+2*pi*(f2(1)*pos(i,1)+f2(2)*pos(i,2)))+a3(2)*cos(fase3+2*pi*(f3(1)*pos(i,1)+f3(2)*pos(i,2))))/ft;

      %v=flujo_radial(pos(i,:),pto_central,v_min,v_max);
      %v=flujo_radial(pos(i,:),pto_central,0.001,0.25)+flujo_circular(pos(i,:),pto_central,0,1);
      %v=flujo_radial(pos(i,:),pto_central,v_min,v_max);
      %v=flujo_sinusoidal(pos(i,:),v0,f1,fase)+[0.6 0.6];
      
      x = xx(i,:);
      
      eval(['v = ',flow,';'])
      
      despx=v(1)/ft;
      despy=v(2)/ft;
      
      d=[d;despx despy];

   end;
   xxant=xx;
   xx=xx+d; 
   speeds(:,:,j-1)=d*ft; 
   
   g = generate_frame(xx,sizes,colors,Nx,Nx,fs,0);
   seq=[seq g];   
end

