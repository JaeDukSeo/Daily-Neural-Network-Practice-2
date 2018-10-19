% ---------------------------------------
% (RANDOM) DOTS SEQUENCES
%
% See additional demos in BasicVideoTools:
%   demo_motion_programs.m 
%   example_rendom_dots_sequence.m
% See additional motion estimation toolbox:
%   http://isp.uv.es/Video_coding.html
% ----------------------------------------
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
% BasicVideoTools comes with functions to implement the non-trivial steps:
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
% And a convenient function to call these in a loop:  dots_sequence.m
%
% Here you have an example on how to use them to generate sequences.
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% INITIALIZATION:
%%   Sequence parameters (sampling frequencies and spatio-temporal size)
%%   Object parameters   (sizes, colors and initial spatial locations)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % SEQUENCE PARAMETERS    
   % Spatial size (in pixels). We assume squared frames.
   % Temporal size (in number of frames).
   % Sampling frequencies (spatial in cycles per degree, and temporal in Hz)
    
        Nx = 120;    % pixels
        Nt = 32;    % frames
        fs = Nx/2;  % This choice implies that the size of the frames is 2 deg.
        ft = 20;

   % PARAMETERS OF THE OBJECTS
   % Number of objects, sizes, colors (only luminance level) and locations.
   %
   % In order to choose these parameters (mainly number and location) you
   % have to consider that the selected flow field (e.g. radial expansive flow) 
   % will be applied to the objects, and hence they may be expelled from
   % the image after some frames. 
   % Similarly, if you plan to apply a translation flow, some objects have
   % to be placed outside the field of view ready to jump in after the 
   % displacements that will happen in some delta_t = 1/ft
   %
    
   % NUMBER
     N_objects = 35000;   % Number of objects

   % SIZES in degrees. Note that the size of the frame in degrees is Nx/fs
   %                   Subpixel sizes (small dots) are fine. Their energy
   %                   will spread out over the pixel with the
   %                   corresponding reduction in amplitude (see details in generate_image.m).
   %

     % Range of sizes (will be random)
     S_max=0.04; % in deg
     S_min=0.03; % in deg

     widths = S_min + (S_max-S_min)*rand(N_objects,1);
    heights = S_min + (S_max-S_min)*rand(N_objects,1);
      sizes = [widths heights];

   % COLOR (gray level)

      Colormax=255;
      Colormin=70;
      colors = Colormin + ceil((Colormax-Colormin)*rand(N_objects,1));

   %  LOCATIONS in degrees. (random in this example):
   %                        Note that the origin of the spatial domain is
   %                        at the top-left corner, and the center of the
   %                        frame is at [Nx/fs Nx/fs]/2

      locations_centered = [(Nx/fs)*rand(N_objects,2) - repmat([Nx/(2*fs) Nx/(2*fs)],N_objects,1)]; % These cover the spatial extension but contain negative numbers
        locations_scaled = 12*locations_centered;                                                    % Apply whatever scale factor you like (bigger if speed is big) 
       initial_locations = locations_scaled + repmat([Nx/(2*fs) Nx/(2*fs)],N_objects,1);            % Re-center

               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% SEQUENCE GENERATION:
%%   Initial data + Flow data
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % Build your flow field using the available elements (see the examples)
     %flow = 'radial_flow(x,[1 1],0.001,0.25) + circular_flow(x,[1 1],0,1) + 0.1*randn(1,2)';
     flow_R = 'radial_flow(x,[1 1],0.05,0.5)';
     flow_L = 'lateral_flow(x,1,3)';
     flow_LN = 'lateral_flow_no_sky(x,1,3)';
     flow_R_L = 'radial_flow(x,[1 1],0.05,0.5)+lateral_flow(x,1,3)';
     flow_R_LN = 'radial_flow(x,[1 1],0.05,0.5)+lateral_flow_no_sky(x,1,3)';
     flow_unif_rand = 'lateral_flow_no_sky(x,20,2/20)+ 0.5*randn(1,2)';
     flow_rand = '0.5*randn(1,2)';

     % flow = 'sinusoidal_flow(x,[0 0.5],[1 1],0);'
     
   % Take the objects, generate the frame, compute speeds, update locations, and repeat
     [sequence_R,locations_R,speeds_R] = dots_sequence(initial_locations,sizes,colors,flow_R,Nx,Nt,fs,ft); 
     [sequence_L,locations_L,speeds_L] = dots_sequence(initial_locations,sizes,colors,flow_L,Nx,Nt,fs,ft); 
     [sequence_LN,locations_LN,speeds_LN] = dots_sequence(initial_locations,sizes,colors,flow_LN,Nx,Nt,fs,ft); 
     [sequence_RL,locations_RL,speeds_RL] = dots_sequence(initial_locations,sizes,colors,flow_R_L,Nx,Nt,fs,ft); 
     [sequence_RLN,locations_RLN,speeds_RLN] = dots_sequence(initial_locations,sizes,colors,flow_R_LN,Nx,Nt,fs,ft); 
     [sequence_unif,locations_unif,speeds_unif] = dots_sequence(initial_locations,sizes,colors,flow_unif_rand,Nx,Nt,fs,ft); 
     [sequence_rand,locations_rand,speeds_rand] = dots_sequence(initial_locations,sizes,colors,flow_rand,Nx,Nt,fs,ft); 

     
   % Build the matlab movie sequence from the data
     MR = build_achrom_movie(sequence_rand,0,255,Nx,1);  
     ML = build_achrom_movie(sequence_L,0,255,Nx,1);  
     MLN = build_achrom_movie(sequence_LN,0,255,Nx,1);  
     MRL = build_achrom_movie(sequence_RL,0,255,Nx,1);
     MRLN = build_achrom_movie(sequence_RLN,0,255,Nx,1);
     Mu = build_achrom_movie(sequence_unif,0,255,Nx,1);
     
     save('C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\propaganda')
          
     O = then2now(sequence_rand,Nx);
     A = then2now(sequence_unif,Nx);
     B = then2now(sequence_R,Nx);
     C = then2now(sequence_RLN,Nx);
     
%      Ob = O;
     Ab = A;
%      Bb = B;
     Cb = C;
     tam = 3;
     for i=1:Nt
         Ob(:,:,i) = 1.2*filter2(ones(tam,tam)/tam^2,O(:,:,i),'same');
         Ab(:,:,i) = 1.2*filter2(ones(tam,tam)/tam^2,A(:,:,i),'same');
         Bb(:,:,i) = 1.2*filter2(ones(tam,tam)/tam^2,B(:,:,i),'same');
         Cb(:,:,i) = 1.2*filter2(ones(tam,tam)/tam^2,C(:,:,i),'same');
     end
          
     save_avi_file(cat(3,O,O(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\N_prop_rand.avi',1)
     save_avi_file(cat(3,A,A(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\N_prop_lateral.avi',1)
     save_avi_file(cat(3,B,B(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\N_prop_R.avi',1)
     save_avi_file(cat(3,C,C(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\N_prop_RLN.avi',1)
     
     save_avi_file(cat(3,Ob,Ob(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\B_prop_randB.avi',1)
     save_avi_file(cat(3,Ab,Ab(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\B_prop_lateralB.avi',1)
     save_avi_file(cat(3,Bb,Bb(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\B_prop_RB.avi',1)
     save_avi_file(cat(3,Cb,Cb(:,:,end-1:-1:1)),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\B_prop_RLNB.avi',1)     

     save_avi_file( cat(2,cat(3,O,O(:,:,end-1:-1:1)), cat(3,Ob,Ob(:,:,end-1:-1:1))),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\NB_prop_rand_NB.avi',1)
     save_avi_file( cat(2,cat(3,A,A(:,:,end-1:-1:1)), cat(3,Ab,Ab(:,:,end-1:-1:1))),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\NB_prop_lateral_NB.avi',1)
     save_avi_file( cat(2,cat(3,B,B(:,:,end-1:-1:1)), cat(3,Bb,Bb(:,:,end-1:-1:1))),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\NB_prop_R_NB.avi',1)
     save_avi_file( cat(2,cat(3,C,C(:,:,end-1:-1:1)), cat(3,Cb,Cb(:,:,end-1:-1:1))),0,255,Nx,'C:\disco_portable\mundo_irreal\jesus\CLASES\sesion_propaganda\NB_prop_RL_NB.avi',1)     
     
     