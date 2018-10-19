% ----------------------------------------
% DEMO OF BASIC VIDEO TOOLS
%
% See additional demos in BasicVideoTools:
%   example_random_dots_sequence.m 
%   example_newtonian_sequences.m
% See additional motion estimation toolbox:
%   http://isp.uv.es/Video_coding.html
%
% ----------------------------------------
%
% In this demo: 
% 
%  * We consider five sequences:
%       . a natural sequence from VQEG: Y_vqeg
%       . a natural sequence from LIVE: Y_live
%       . one sequence obtained by moving a natural image with uniform speed: Y1_2d 
%       . one sequence obtained by moving colored noise with uniform speed:   Y2_2d 
%       . a natural sequence with illustrative motion:                        Y3_2d
%         (the variable "Y" in file "real_seq_Nx_75_Nf_64.mat" will be named Y3_2d)
%    The VQEG and LIVE sequences are skipped if the database was not downloaded. 
%
%  * We analyze the PCA of patches from the last three sequences.
%
%  * We analyze the Fourier transform of the last three sequences. 
%
%  * We compute the response of a set of V1 cells tuned to a range of spatial 
%    positions (covering the whole spatial domain) and to illustrative 
%    spatio-temporal frequencies (useful for the last three sequences).
%
%  * We compute the response of a set of MT cells tuned to a range of spatial 
%    positions (covering the whole spatial domain) and to illustrative 
%    spatio-temporal frequencies (useful for the last three sequences).
%
%  * We compute the CSF filtered version of the last sequence
%
%  * We generate a sequence to illustrate the Waterfall after-effect
%

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%  
%%  SEQUENCES
%%  
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

%
% Read original sequences from the VQEG and LIVE databases 
% (distorted sequences are not distributed in this toolbox)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(' ')
disp(' ')
disp(' Please enter a string with the folder where you stored the video data ')
disp(' ...for instance    /media/disk/vista/Software/video_software/ ')
disp(' If you did not downloaded the optional video data, enter 0')
toolbox_folder = input(' ');

if toolbox_folder ~= 0

    addpath(genpath(toolbox_folder))

    % VQEG
    Y_vqeg = read_vqeg_mg([toolbox_folder 'video_data/VQEG/src3_ref__625.yuv']);

    % LIVE
    Y_live = read_live_mg([toolbox_folder 'video_data/LIVE/bs1_25fps.yuv']);

    % Matlab videoplayer to visualize 3d arrays (slower than Matlab movie function but with higher functionalities)

    try
        implay(Y_vqeg/255)   % CAUTION!: this viewer is available after image processing toolbox 2010
        implay(Y_live/255)
    catch
        fig = 1;
        s = size(Y_vqeg);
        M_vqeg = build_achrom_movie(Y_vqeg, 0, 255, s(2), fig);
        s = size(Y_live);
        M_live = build_achrom_movie(Y_live, 0, 255, s(2), fig+1);
    end

    clear M_vqeg M_live Y_vqeg Y_live
end
        %
        % READ GENERAL LIVE AND VQEG VIDEOS: COLOR AND DISTORTED MOVIES (not provided in this toolbox)
        %

        %% For VQEG: read_vgeg  
        %% -------------------
        %% Returns the three YUV components in sequential frames of the array.
        %% Therefore the 1:3:end subsampling to get the luminance component.
        %% If color data were required one should use ycbcr2rgb to get the true
        %% color images in RGB
        %
        %   [YUV_orig, YUV_dist] = read_vqeg(path,ind_mov,want_orig,ind_dist);
        %   Y_orig = YUV_orig(:,:,1:3:end);
        %
        %% For LIVE: yuv2mov plus loop
        %% ---------------------------
        %% Returns a Matlab movie structure (with data in RGB). 
        %% The loop below extracts the luminance component of each frame and stores
        %% it in the n-dimensional array
        %
        %    mov = yuv2mov('name.yuv', width, height);
        %    Y_orig = zeros(height, width, length(mov));
        %    for ii=1:length(mov)
        %         frame = data(ii).cdata;
        %         frame = rgb2ycbcr(frame);
        %         Y_orig(:,:,ii) = double(squeeze(frame(:,:,1)));
        %    end


% Generate sequence of controlled speed from still image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  In this example first I move the image down and then up


    %    % Matlab gatlin image
    %    %
    %    % Still image is the variable X (indexed image) in the file "gatlin" 
    %    % Luminances are in the range [1,64];
    %    load gatlin
    %    
    %    center1 = [100 80]; % Center (row,col) of first frame in the still image
    %    Nx = 75;            % Size of frame
    %    Nf = 32;            % Number of frames
    %    fsx = 37.5;         % Spatial sampling frequency (in cpd)
    %    fst = 24;           % Temporal sampling frequency (in Hz)
    %    
    %    v = [0.875 0];      % Selected speed [vy vx] (in deg/sec)
    %    interpolat = 1;     % Interpolate if the selected speed leads to subpixel displacements.
    % 
    %    % This is the first part of the sequence (moving to the right) in the (old) 2d array format
    %    [Y1 , v_true] = image_sequence(X,center1,Nx,Nf,fsx,fst,v,interpolat);
    %    
    %    % Now I rearrange the Y1 as a 3d array to easily reverse frames. Then I
    %    % build the back and forth movie
    %    
    %    Y1_3d = then2now(Y1,Nx);
    %    Y1_3d_reverse = Y1_3d(:,:,end:-1:1);
    %    
    %    Y1_3d_forward_and_backward = cat(3,Y1_3d,Y1_3d_reverse);
    %    
    %    Y1_2d = now2then(Y1_3d_forward_and_backward);
    %    clear Y1_3d_forward_and_backward Y1_3d Y1_3d_reverse          
    %    Ymin = 1;
    %    Ymax = 64;
    %    fig=1;
    %    M1 = build_achrom_movie(Y1_2d, Ymin, Ymax, Nx, fig);          
   
   
   % Calibrated Luminance image from the Barcelona (CVC) database
   %
   % Still image is the variable Y in the file Y_i.mat (with i=1:418) in the Barcelona_luminance folder;
   %
   
   load Y_3.mat
   
   center1 = [380 500]; % Center (row,col) of first frame in the still image
   Nx = 75;            % Size of frame
   Nf = 32;            % Number of frames
   fsx = 37.5;         % Spatial sampling frequency (in cpd)
   fst = 24;           % Temporal sampling frequency (in Hz)
   
   v = [0.875 0];      % Selected speed [vy vx] (in deg/sec)
   interpolat = 1;     % Interpolate if the selected speed leads to subpixel displacements.

   % This is the first part of the sequence (moving to the right) in the (old) 2d array format
   [Y1 , v_true] = image_sequence(Y.^0.5,center1,Nx,Nf,fsx,fst,v,interpolat);
   
   % Now I rearrange the Y1 as a 3d array to easily reverse frames. Then I
   % build the back and forth movie
   
   Y1_3d = then2now(Y1,Nx);
   Y1_3d_reverse = Y1_3d(:,:,end:-1:1);
   
   Y1_3d_forward_and_backward = cat(3,Y1_3d,Y1_3d_reverse);
   
   Y1_2d = now2then(Y1_3d_forward_and_backward);
   clear Y1_3d_forward_and_backward Y1_3d Y1_3d_reverse          
   Ymin = min(Y1_2d(:));
   Ymax = max(Y1_2d(:));
   fig=1;
   M1 = build_achrom_movie(Y1_2d, Ymin, Ymax, Nx, fig); 
  
   
% Generate sequence of controlled speed from noise image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  In this example first I move the noise to the right and then to the left   
% . Colored noise sequences of controlled speed

          % Parameters of the colored noise
          fmin = 2;
          fmax = 12;
          orient = 0;
          delta_orient = 180;
          
          % Speed of the noise
          v = [1 0];       % Selected speed [vy vx] (in deg/sec)
          interpolat = 1;  % Interpolate if the selected speed leads to subpixel displacements.          
          
  [Y2 , v_true] = noise_sequence( Nx , Nf , fsx , fst , fmin , fmax , delta_orient , orient , v, interpolat );
  
  % Normalize the amplitude of the noise so that its luminance is in the
  % same range as the regular sequences
  
  Y2 = Ymax*(Y2 - min(min(Y2)))/(max(max(Y2)) - min(min(Y2)));
  
   % Now I rearrange the Y1 as a 3d array to easily reverse frames. Then I
   % build the back and forth movie
   
   Y2_3d = then2now(Y2,Nx);
   Y2_3d_reverse = Y2_3d(:,:,end:-1:1);
   
   Y2_3d_forward_and_backward = cat(3,Y2_3d,Y2_3d_reverse);
   
   Y2_2d = now2then(Y2_3d_forward_and_backward);
   clear Y2_3d_forward_and_backward Y2_3d Y2_3d_reverse
   M2 = build_achrom_movie(Y2_2d, 0, Ymax, Nx, fig+1);          

% Natural sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load real_seq_Nx_75_Nf_64.mat
Y3_2d = Y;
Y3_2d = Ymax*(Y3_2d - min(min(Y3_2d)))/(max(max(Y3_2d)) - min(min(Y3_2d)));

M3 = build_achrom_movie(Y3_2d, 0, Ymax, Nx, fig+2);

% All together (just for fun)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rearrange data
Y1 = then2now(Y1_2d,Nx);
Y2 = then2now(Y2_2d,Nx);
Y3 = then2now(Y3_2d,Nx);

% Generate big movie from sub-movies
Y = zeros(Nx,3*Nx,length(Y1(1,1,:)));
for i=1:length(Y1(1,1,:))
    Y(:,:,i) = [squeeze(Y1(:,:,i)) squeeze(Y2(:,:,i)) squeeze(Y3(:,:,i))]; 
end

M = build_achrom_movie(Y, 0, Ymax, 3*Nx, fig+3);
figure(fig+3),movie(M,5,fst)

clear Y M

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%  
%%  PCA
%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

   % Extract data patches and rearrange the patches in vectors 

    N = 10;   % Spatial size of the patch
    Nf = 15;  % Temporal length of the patch
 
    % From the first sequence (a lot more data may be extracted):
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    
       x1 = im2colcube(Y1(:,:,1:Nf),[N N],2);
       x1 = x1(:,1:4:end);
       for i = 1:5:length(Y1(1,1,:))-Nf-1;
           i
           aux = im2colcube(Y1(:,:,1+i:Nf+i),[N N],2);
           x1 = [x1 aux(:,1:4:end)];
       end;
       
    % Approximate covariance matrix (small training set and approximate mean vector)
       Y_mean = mean(mean(x1));  % I assume uniform mean patch to skip the repmat part to get zero mean
       x1 = x1-Y_mean;
       C1 = x1*x1'/length(x1(1,:));
       clear x1;
       
    % PCA
       tic,[B1,D1] = eigs(C1,256);toc
    
       figur = 100;
       number_funct = 12;
       [MB1,array] = disp_spatio_temp_patches(B1(:,1:number_funct^2),number_funct,N,Nf,figur);
       figure,semilogy(diag(D1))

    % From the third sequence (a lot more data may be extracted):
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    
       x1 = im2colcube(Y3(:,:,1:Nf),[N N],2);
       x1 = x1(:,1:4:end);
       for i = 1:5:length(Y3(1,1,:))-Nf-1;
           i
           aux = im2colcube(Y1(:,:,1+i:Nf+i),[N N],2);
           x1 = [x1 aux(:,1:4:end)];
       end;
       
    % Approximate covariance matrix (small training set and approximate mean vector)
       Y_mean = mean(mean(x1));  % I assume uniform mean patch to skip the repmat part to get zero mean
       x1 = x1-Y_mean;
       C3 = x1*x1'/length(x1(1,:));
       clear x1;
       
    % PCA
       tic,[B3,D3] = eigs(C3,256);toc
    
       figur = 101;
       number_funct = 12;
       [MB3,array] = disp_spatio_temp_patches(B3(:,1:number_funct^2),number_funct,N,Nf,figur);
       figure,semilogy(diag(D3))
       
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%  
%%  FOURIER TRANSFORM OF SEQUENCES
%%  
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

% Temporal window to reduce the edge effects of filtering (5% of the movie length)
Nf = length(Y3(1,1,:));
[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(Nx,Nx,Nf,fsx,fsx,fst);
delta_t = 0.05*(Nf/fst);  
temp_w = temporal_window(t,delta_t);

% Computation
do_fftshift =1;
TF1 = fft3(Y1_2d-mean(mean(Y1_2d)),do_fftshift);
TF2 = fft3(Y2_2d-mean(mean(Y2_2d)),do_fftshift);
TF3 = fft3(temp_w.*(Y3_2d-mean(mean(Y3_2d))),do_fftshift); % I apply the window only here because this is the sequence I am going to use afterwards

% Display
fig=10;
show_fft3( abs(TF1).^(1/3), fsx, fst, fig);title('Gatlin')
show_fft3( abs(TF2).^(1/2), fsx, fst, fig+1);title('Noise')
show_fft3( abs(TF3).^(1/3), fsx, fst, fig+2);title('Natural')

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%  
%%  V1 and MT cells, spatio-temporal CSF
%%  
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

% Sets of V1 cells (Receptive Fields in the Fourier domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    delta_fx=2;
    delta_fy=2;
    delta_ft=1;
    
    fxo=0;
    fyo=7.5;
    fto=[1 6.5 12]; % From low to high frequency (low to high speed)

    % Computation (tuned to diferent frequencies)
    G_V1_1 = sens_gabor3d(Nx,Nx,Nf,fsx,fsx,fst,fxo,fyo,fto(1),delta_fx,delta_fy,delta_ft);
    G_V1_2 = sens_gabor3d(Nx,Nx,Nf,fsx,fsx,fst,fxo,fyo,fto(2),delta_fx,delta_fy,delta_ft);   
    G_V1_3 = sens_gabor3d(Nx,Nx,Nf,fsx,fsx,fst,fxo,fyo,fto(3),delta_fx,delta_fy,delta_ft);   
    G_V1_m1 = sens_gabor3d(Nx,Nx,Nf,fsx,fsx,fst,fxo,fyo,-fto(1),delta_fx,delta_fy,delta_ft);   
    G_V1_m2 = sens_gabor3d(Nx,Nx,Nf,fsx,fsx,fst,fxo,fyo,-fto(2),delta_fx,delta_fy,delta_ft);   
    G_V1_m3 = sens_gabor3d(Nx,Nx,Nf,fsx,fsx,fst,fxo,fyo,-fto(3),delta_fx,delta_fy,delta_ft);   

    % Normalization
    G_V1_1 = G_V1_1/sum(sum(G_V1_1));
    G_V1_2 = G_V1_2/sum(sum(G_V1_2));
    G_V1_3 = G_V1_3/sum(sum(G_V1_3));
    G_V1_m1 = G_V1_m1/sum(sum(G_V1_m1));
    G_V1_m2 = G_V1_m2/sum(sum(G_V1_m2));
    G_V1_m3 = G_V1_m3/sum(sum(G_V1_m3));
    
    % Show transform
    show_fft3( G_V1_1, fsx, fst, fig+10),title('Neurons V1 1')
    show_fft3( G_V1_2, fsx, fst, fig+11),title('Neurons V1 2')
    show_fft3( G_V1_3, fsx, fst, fig+12),title('Neurons V1 3')
    show_fft3( G_V1_m1, fsx, fst, fig+13),title('Neurons V1 -1')
    show_fft3( G_V1_m2, fsx, fst, fig+14),title('Neurons V1 -2')
    show_fft3( G_V1_m3, fsx, fst, fig+15),title('Neurons V1 -3')
    
% Sets of MT cells (Receptive Fields in the Fourier domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Computation (tuned to diferent speed)
    G_MT_1 = sens_MT(Nx,Nx,Nf,fsx,fsx,fst,[0 0.15]);
    G_MT_2 = sens_MT(Nx,Nx,Nf,fsx,fsx,fst,[0 0.875]);
    G_MT_3 = sens_MT(Nx,Nx,Nf,fsx,fsx,fst,[0 1.5]);    
    G_MT_m1 = sens_MT(Nx,Nx,Nf,fsx,fsx,fst,[0 -0.15]);
    G_MT_m2 = sens_MT(Nx,Nx,Nf,fsx,fsx,fst,[0 -0.875]);
    G_MT_m3 = sens_MT(Nx,Nx,Nf,fsx,fsx,fst,[0 -1.5]);    

    % Normalization
    G_MT_1 = G_MT_1/sum(sum(G_MT_1));
    G_MT_2 = G_MT_2/sum(sum(G_MT_2));
    G_MT_3 = G_MT_3/sum(sum(G_MT_3));
    G_MT_m1 = G_MT_m1/sum(sum(G_MT_m1));
    G_MT_m2 = G_MT_m2/sum(sum(G_MT_m2));
    G_MT_m3 = G_MT_m3/sum(sum(G_MT_m3));
    
    % Show frequency bands
    show_fft3( G_MT_1, fsx, fst, fig+16),title('Neurons MT 1')
    show_fft3( G_MT_2, fsx, fst, fig+17),title('Neurons MT 2')
    show_fft3( G_MT_3, fsx, fst, fig+18),title('Neurons MT 3')
    show_fft3( G_MT_m1, fsx, fst, fig+19),title('Neurons MT -1')
    show_fft3( G_MT_m2, fsx, fst, fig+20),title('Neurons MT -2')
    show_fft3( G_MT_m3, fsx, fst, fig+21),title('Neurons MT -3')

% Response of cells (normalize to the maximum response for visualization)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Computation v1
r_v1_1_seq3 = real(ifft3( TF3.*sqrt(G_V1_1) , 1));
r_v1_2_seq3 = real(ifft3( TF3.*sqrt(G_V1_2) , 1));
r_v1_3_seq3 = real(ifft3( TF3.*sqrt(G_V1_3) , 1));
r_v1_m1_seq3 = real(ifft3( TF3.*sqrt(G_V1_m1) , 1));
r_v1_m2_seq3 = real(ifft3( TF3.*sqrt(G_V1_m2) , 1));
r_v1_m3_seq3 = real(ifft3( TF3.*sqrt(G_V1_m3) , 1));

% Normalization
mv1 = min(min([r_v1_1_seq3 r_v1_2_seq3 r_v1_3_seq3 r_v1_m1_seq3 r_v1_m2_seq3 r_v1_m3_seq3]));
Mv1 = max(max([r_v1_1_seq3 r_v1_2_seq3 r_v1_3_seq3 r_v1_m1_seq3 r_v1_m2_seq3 r_v1_m3_seq3]));
r_v1_1_seq3 = (r_v1_1_seq3 - mv1)/(Mv1-mv1);
r_v1_2_seq3 = (r_v1_2_seq3 - mv1)/(Mv1-mv1);
r_v1_3_seq3 = (r_v1_3_seq3 - mv1)/(Mv1-mv1);
r_v1_m1_seq3 = (r_v1_m1_seq3 - mv1)/(Mv1-mv1);
r_v1_m2_seq3 = (r_v1_m2_seq3 - mv1)/(Mv1-mv1);
r_v1_m3_seq3 = (r_v1_m3_seq3 - mv1)/(Mv1-mv1);

% Computation MT
r_mt_1_seq3 = real(ifft3( TF3.*sqrt(G_MT_1) , 1));
r_mt_2_seq3 = real(ifft3( TF3.*sqrt(G_MT_2) , 1));
r_mt_3_seq3 = real(ifft3( TF3.*sqrt(G_MT_3) , 1));
r_mt_m1_seq3 = real(ifft3( TF3.*sqrt(G_MT_m1) , 1));
r_mt_m2_seq3 = real(ifft3( TF3.*sqrt(G_MT_m2) , 1));
r_mt_m3_seq3 = real(ifft3( TF3.*sqrt(G_MT_m3) , 1));

% Normalization
mmt = min(min([r_mt_1_seq3 r_mt_2_seq3 r_mt_3_seq3 r_mt_m1_seq3 r_mt_m2_seq3 r_mt_m3_seq3]));
Mmt = max(max([r_mt_1_seq3 r_mt_2_seq3 r_mt_3_seq3 r_mt_m1_seq3 r_mt_m2_seq3 r_mt_m3_seq3]));
r_mt_1_seq3 = (r_mt_1_seq3 - mmt)/(Mmt-mmt);
r_mt_2_seq3 = (r_mt_2_seq3 - mmt)/(Mmt-mmt);
r_mt_3_seq3 = (r_mt_3_seq3 - mmt)/(Mmt-mmt);
r_mt_m1_seq3 = (r_mt_m1_seq3 - mmt)/(Mmt-mmt);
r_mt_m2_seq3 = (r_mt_m2_seq3 - mmt)/(Mmt-mmt);
r_mt_m3_seq3 = (r_mt_m3_seq3 - mmt)/(Mmt-mmt);

% Montage of a nice illustrative movie of responses in V1 and MT
Y3_3d = then2now(Y3_2d,Nx);
Y3_3d = Y3_3d/max(max(Y3_2d));

r_v1_1_s3 = then2now(r_v1_1_seq3,Nx);
r_v1_2_s3 = then2now(r_v1_2_seq3,Nx);
r_v1_3_s3 = then2now(r_v1_3_seq3,Nx);
r_v1_m1_s3 = then2now(r_v1_m1_seq3,Nx);
r_v1_m2_s3 = then2now(r_v1_m2_seq3,Nx);
r_v1_m3_s3 = then2now(r_v1_m3_seq3,Nx);
r_mt_1_s3 = then2now(r_mt_1_seq3,Nx);
r_mt_2_s3 = then2now(r_mt_2_seq3,Nx);
r_mt_3_s3 = then2now(r_mt_3_seq3,Nx);
r_mt_m1_s3 = then2now(r_mt_m1_seq3,Nx);
r_mt_m2_s3 = then2now(r_mt_m2_seq3,Nx);
r_mt_m3_s3 = then2now(r_mt_m3_seq3,Nx);

RV1 = zeros(2*Nx,4*Nx,length(r_v1_1_s3(1,1,:)));
RMT = RV1;
for i=1:length(r_v1_1_s3(1,1,:))
    foto1 = [Y3_3d(:,:,i) r_v1_1_s3(:,:,i) r_v1_2_s3(:,:,i) r_v1_3_s3(:,:,i);...
           0*Y3_3d(:,:,i) r_v1_m1_s3(:,:,i) r_v1_m2_s3(:,:,i) r_v1_m3_s3(:,:,i)];
    foto2 = [Y3_3d(:,:,i) r_mt_1_s3(:,:,i) r_mt_2_s3(:,:,i) r_mt_3_s3(:,:,i);...
           0*Y3_3d(:,:,i) r_mt_m1_s3(:,:,i) r_mt_m2_s3(:,:,i) r_mt_m3_s3(:,:,i)];
     
    RV1(:,:,i)=foto1;
    RMT(:,:,i)=foto2;    
end

M_V1 = build_achrom_movie(RV1,0,1,4*Nx,200);
M_MT = build_achrom_movie(RMT,0,1,4*Nx,1);
% resp = build_achrom_movie(P,0,1,4*Nx,1001);

P = cat(1,RV1,RMT);
P(150:end,1:75,:) = zeros(size(P(150:end,1:75,:)));

% Here is the movie with the sequence and the responses

try
  implay(P)
catch
   fig = 200; 
   s = size(P); 
   M_resp = build_achrom_movie(P, 0, 1, s(2), fig);
   figure(fig),movie(M_resp,5,12);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CSF filtering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stabiliz = 0; % No stabilization of ocular motion
fy0 = 0;
[csfet,csf_fx_ft,fx,ft] = spatio_temp_csf(fsx,fsx,fst,Nx,Nx,Nf,stabiliz,fy0);

r_csf_2d = real(ifft3( TF3.*csfet , 1));
m = min(r_csf_2d(:));
M = max(r_csf_2d(:));
r_csf_3d = then2now(r_csf_2d,Nx);
r_csf_3d = (r_csf_3d-m)/(M-m);

m = min(Y3(:));
M = max(Y3(:));
Y3 = (Y3-m)/(M-m);

seq_csf = cat(2,then2now(temp_w,Nx).*Y3,r_csf_3d);

try
  implay(seq_csf)
catch
   fig = 200; 
   s = size(seq_csf); 
   % M_resp = build_achrom_movie(seq_csf, 0, 1, s(2), fig);
   S_CSF = build_achrom_movie(seq_csf,0,1,s(2),200);
   figure(fig),movie(S_CSF,5,12);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%  
%%  Static Motion After-effect (Waterfall effect)
%%  
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

% Domain parameters
ne=40;
nt=200;
fse=40;
fst=20;

% Noise parameters
fm=0;
fM=fse/5;
anch_ang=179.9;
orient=0;
v=[-0.2 0;-2.5 0];

% Generate two sequences with different speed and merge them

for vv =1:2

    Lm=128;
    C=1;

    % Ruido
    [s,v_real] = noise_sequence(ne,nt,fse,fst,fm,fM,anch_ang,orient,v(vv,:),1);
    S=Lm*(1+2*C*(s-0.5));

    cond=ones(40,40);
    cond(18:22,20)=zeros(5,1);
    cond(20,18:22)=zeros(1,5);

    S=S/4;
    for i=1:nt
        fotog=sacafot(S,ne,ne,i);
        fotog=fotog.*cond;
        S=metefot(S,fotog,i,1);
    end

    eval(['S',num2str(vv),'=S;'])

end

for i=1:nt
    fotog1=sacafot(S1,ne,ne,i);
    fotog2=sacafot(S2,ne,ne,i);
    fotog = [fotog1(:,1:ne/2) fotog2(:,ne/2+1:end)];
    S=metefot(S,fotog,i,1);
end

SS = [S repmat(fotog,1,80)];

% Build the movie from SS
P = build_achrom_movie(S,0,(2*Lm)/4,ne,1);

% Visualizacion de la peli

figure(1),movie(P,1,fst)
