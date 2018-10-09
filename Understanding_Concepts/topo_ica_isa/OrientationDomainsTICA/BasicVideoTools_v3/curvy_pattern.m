function g_filt = curvy_pattern(Nx,fsx,fc1,fc2,orient,anch_ang,fcc1,fcc2)

%
%  Nx = 384;
%  fsx = 64;
%  fc = 1*[0 1];
%  fcc = 1*[2 4];
%  orient = 0;
%  orient_width = 30;  % Caution! if fc is small orientation width has to
%                      %          be wide enough to ensure non-zero band in
%                      %          the discrete frerquency domain
%
%  g_filt = curvy_pattern(Nx,fsx,fc(1),fc(2),orient,orient_width,fcc(1),fcc(2));

[xx,yy,tt,fxx,fyy,ftt] = spatio_temp_freq_domain(Nx,Nx,1,fsx,fsx,1);


% fc cpd noise
%%%%%%%%%%%%%%%%%%%%%%

%  fc1 = fc(1);
%  fc2 = fc(2);
%  
%  filtro_freq = (fxx.^2 + fyy.^2) > fc1^2 & (fxx.^2 + fyy.^2) < fc2^2;
% 
%  N_freq = filtro_freq.*exp(i*2*pi*rand(size(xx))) ;
%  
%  n_freq = real(ifft2(fftshift(N_freq)));
%             % figure,colormap gray,imagesc(xx(1,:),xx(1,:),n_freq)
            
            
[ss,fil,s1]=rcolor2d(fsx,Nx,fc1,fc2,orient,anch_ang,1,1);            
 
n_freq = s1 - mean(s1(:));
  
% binary
%%%%%%%%%%%%%%%%%%%%

 n_freq_bin = double(n_freq>0);
            %figure,colormap gray,imagesc(xx(1,:),xx(1,:),n_freq_bin)

% edge detection (white)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h = [0 -0.25 0;-0.25 1 -0.25;0 -0.25 0];

im_bordes = conv2(n_freq_bin,h,'same');

% gusanito black
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % figure,colormap gray,imagesc(xx(1,:),xx(1,:),abs(im_bordes)==0)

g_black = double(abs(im_bordes)==0);
 
% filtra a fcc cpd
%%%%%%%%%%%%%%%%%%%%%%%%%%

%  fcc1 = fcc(1);
%  fcc2 = fcc(2);
 
 filtro_freq = (fxx.^2 + fyy.^2) > fcc1^2 & (fxx.^2 + fyy.^2) < fcc2^2;

 %N_freq = filtro_freq.*exp(i*2*pi*rand(size(xx))) ;
 
 
 g_filt = real(ifft2(ifftshift(filtro_freq.*fftshift(fft2(g_black)))));
 
       % figure,colormap gray,imagesc(xx(1,:),xx(1,:),g_filt)
  