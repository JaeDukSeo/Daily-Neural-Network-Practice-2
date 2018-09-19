%
% --------------------------------------
% BASIC VIDEO TOOLS (version 3)
% Jesus Malo            jesus.malo@uv.es 
% Juan Gutierrez
% Valero Laparra
% (c) Universitat de Valencia. 1996-2016
% --------------------------------------
%
% Read video data: from files to 3D arrays (height*width*frames)
% ---------------
%
%    read_vqeg_mg - Simplified function to read achromatic part of original VQEG movies    
%    read_live_mg - Simplified function to read achromatic part of original LIVE movies
%    yuv2mov   - read yuv format movies (e.g. LIVE) and convert them to Matlab movies
%    read_vqeg - read VQEG files (both original and distorted)
%
% Rearrange data: 
% ---------------
%
%    im2colcube - From 3D arrays (or 3D patches) to vectors
%    col2imcube - From vectors to 3D arrays
%    now2then   - From 3D arrays (modern format) to 2D arrays (old film-like format)
%    then2now   - from 2D arrays to 3D arrays
%
% Generate controlled sequences:
% ------------------------------ 
%    
%    spatio_temp_freq_domain - Define spatio-temporal domains to build analytical sequences therein
%    control_lum_contrast    - Imposes certain average luminance and std-contrast in an achromatic natural image
%    combine_test_background - combines two controlled images in a center-surround setting 
%    image_sequence          - Sequences from still images moving at controlled speed
%    noise_sequence          - Sequences from colored noise of controlled speed
%    dots_sequence           - Sequences from a set of objects and a flow field 
%       radial_flow          - Computes flow for ego-motion parallel to optical axis
%       lateral_flow         - Computes flow for ego-motion orthogonal to optical axis
%       circular_flow        - Circular flow
%       sinusoidal_flow      - Motion grating 
%    newtonian_sequence      - Sequence from a rigid solid moving in force fields
%       elipso3              - Definition of facets and trapezoids of an ellipsoid
%       dinam_tr             - Translation and rotation newtonian dynamics (Runge-Kutta integration) 
%       pintael2             - Illumination of facets and projection onto the camera
%
% Movie visualization (achromatic only):
% --------------------------------------
%
%    build_achrom_movie_avi -  from 3D and 2D arrays to Matlab movies and avi files
%    disp_spatio_temp_patches - Displays N^2 spatiotemporal patches at the same time
%    disp_patches - Displays N^2 spatial patches at the same time
%    disp_patches_norm - Displays N^2 spatial patches at the same time (controlled normalization)
%    implay (not a part of this toolbox): opens an interactive video player
%
% 3D Fourier transforms
% ---------------------
%    
%    fft3               - Computation of spatio-temporal FFT
%    ifft3              - Inverse FFT
%    show_fft3          - Visualization
%    show_project_fft3  - Visualization of integrals
%    temporal_window    - Convenient to reduce edge effects in spatio-temporal filtering
% 
% Perception-related functions
% ----------------------------
%
%    spatio_temp_CSF    - Spatio-temporal CSF (D.H. Kelly JOSA 79)
%    csfsso             - Achromatic CSF of the Standard Spatial Observer (Watson & Malo IEEE ICIP 02)
%    csf_chrom          - Red-Green & Yellow-Blue chromatic CSFs (Mullen Vis. Res. 85)
%    sens_lgn3d_space   - Linear LGN filters (Difference of Gaussians and derivatives of Gaussians, Cai et al. J. Neurophysiol. 97)
%    sens_gabor3d_space - Linear V1 filters (spatio-temporal Gabors, Simoncelli & Heeger Vis.Res. 98)
%    sens_gabor3d_freq  - Linear V1 filters (defined in the frequency domain)
%    sens_MT            - Linear MT filters (coherent combinations of 3D Gabors in the freq. domain, Simoncelli & Heeger Vis.Res. 98).
%
% Motion estimation and compensation (not in this toolbox)
% ----------------------------------
%
%    See:   http://isp.uv.es/Video_coding.html
%           http://www.scholarpedia.org/article/Optic_flow
%
% Demos
% -----
%
%    demo_motion_programs         - Demo on how to use the above functions (except random dots and newtonian sequences)
%    example_random_dots_sequence - Demo on random dots sequences with controlled flow
%    example_newtonian_sequence   - Demo on physics-controlled sequences
%

% Modifications from previous versions
%%%%%%%%%%%%%%%%%

%    csfsso 

%    csf_chrom

%    disp_patches

%    disp_spatio_temp_patches

%    sens_gabor3d_space

%    control_lum_comtrast

%    build_achrom_movie_avi

%    sens_MT

%    control_lum_contrast

%    noise_sequence

%    combine_test_background

%    spatial_csf_speed
