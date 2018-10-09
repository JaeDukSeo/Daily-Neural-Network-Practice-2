function [CSFet,csf_fx_ft,fxx,ftt]=spatio_temp_csf(fsx,fsy,fst,Nx,Ny,Nt,estab,fy0);

%
% SPATIO_TEMP_CSF computes the spatio-temporal CSF of Kelly with or without
% eye movement stabilization in the 3D Fourier domain defined by the
% sampling parameters defined by the user.
%
% The program returs:
%  (1) Data of the 3d CSF in 2d array format.
%
%  (2) The 2d values of the 3D function in the fx,ft plane (for a certain 
%      value of fy=fy0) and the 1d variables fx and ft to represent this plane
%      with imagesc.
%
% SYNTAX:  [csfet,csf_fx_ft,fx,ft] = spatio_temp_csf(fsx,fsy,fst,Nx,Ny,Nt,stabiliz?,fy0);
%
%   INPUT: 
% 
%     fsx = Spatial sampling frequency in cycles/degree in the x direction (columns) 
%     fsy = Spatial sampling frequency in cycles/degree in the y direction (rows)  
%     fst = Temporal sampling frequency in Hz (frames/sec) 
%     Nx  = number of columns in each frame
%     Ny  = number of rows in each frame 
%     Nt  = number of frames
%     stabiliz? = 0, means natural viewing conditions
%               = 1, means compensation of eye movements (reduction of sensitivity in ft=0 Hz)
%     fy0 = selected frequency to cut the 3D CSF to get a 2D (1d-spatial 1d-temporal) function for visualization purposes 
%
%   OUTPUT:
%
%     csfet     = The actual 3D CSF (in 2D format) to be applied to the Fourier transform of sequences 
%     csf_fx_ft = The 2D cut of the 3D CSF for visualization purposes only
%     fx,ft     = Discrete Fourier domain to represent csf_fx_ft
%                 figure,imagesc(ft,fx,csf_fx_ft)
%
%   EXAMPLE OF CSF FILTERING:
%
%    % Load sequence
%         load real_seq_Nx_75_Nf_64.mat
%         Nx = 75;            % Size of frames
%         Nt = 64;            % Number of frames
%         fsx = 37.5;         % Spatial sampling frequency (in cpd)
%         fst = 24;           % Temporal sampling frequency (in Hz)
%
%    % Normalization between [0,Ymax]
%         Ymax = 100;
%         Y_2d = Y;
%         Y_2d = Ymax*(Y_2d - min(min(Y_2d)))/(max(max(Y_2d)) - min(min(Y_2d)));
%
%    % Build a "movie" variable to be displayed with movie 
%    % (not strctly necessary for visualization, you can also use "implay" if the 2D format of the sequence is turned into 3D using then2now.m)
%         Nx = 75; % Number of columns (it is 75 in this sequence)
%         fig = 1; % Figure where the sequence will be displayed
%         M = build_achrom_movie(Y_2d, 0, Ymax, Nx, fig );
%
%    % 3D Fourier transform of the sequence
%         do_fftshift =1;  % This is necessary since the output of spatio_temp_csf has the zero frequency in the center
%         TF = fft3(Y_2d,do_fftshift);
%         show_fft3( abs(TF).^(1/4), fsx, fst, fig+1);title('TF Natural Sequence')
%
%    % CSF
%        fsy=fsx;      % In this sequence the sampling frequency is the same in both directions
%        Ny=Nx;        % In this sequence frames are squared
%        stabiliz=0;   % Natural viewing
%        fy0=0;        % 
%        [csfet,csf_fx_ft,fx,ft] = spatio_temp_csf(fsx,fsy,fst,Nx,Ny,Nt,stabiliz,fy0); 
%
%    % APPLY CSF 
%         r_csf_2d = real(ifft3( TF.*csfet , do_fftshift));
%
%    % Normalization for nicer plot
%          m = min(r_csf_2d(:));
%          M = max(r_csf_2d(:));
%
%    % Build a "movie" variable to be displayed with movie 
%    % (not strctly necessary for visualization, you can also use "implay" if the 2D format of the sequence is turned into 3D using then2now.m)
%         fig = 3; % Figure where the sequence will be displayed
%         M_filt = build_achrom_movie((r_csf_2d-m)/(M-m), 0, 1, Nx, fig+2 );
%

% seq_csf = cat(2,then2now(temp_w,Nx).*Y3,r_csf_3d);

[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(Ny,Nx,Nt,fsx,fsy,fst);

for i=1:Nt
    f=sacafot(ft,Ny,Nx,i);
    ftt(i)=f(1,1);
end

F=sqrt(fx.^2+fy.^2);

ft=abs(ft)+0.0000000000001;
F=F+0.0000000000001;

if estab==1
   CSFet=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);
else
   ft=abs(ft)+0.1*F;
   CSFet=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);    
end    

[m,I]=min( abs(fy(:,1)-fy0) );

csf_fx_ft=zeros(Nx,Nt);
for i=1:Nx
    csf_fx_ft(i,:)=slineat(CSFet,[I i]);
end 

fxx=fx(1,1:Nx);