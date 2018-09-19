function [h,CSFSSO,CSFT,OE]=csfsso(fs,N,g,fm,l,s,w,os)

% CSFSSO computes the CSF filter of the Standard Spatial Observer
% It includes the CSF expression of Tyler and the oblique effect.
%
%        CSFSSO(fx,fy)=CSFTYL(f)*OE(fx,fy)  
%
% where:
%
%        CSFTYL(f) = g*(exp(-(f/fm))-l*exp(-(f^2/s^2)))
%        OE(fx,fy) = 1-w*(4(1-exp(-(f/os)))*fx^2*fy^2)/f^4)
%
%        (fx,fy) = 2D spatial frequency vector (in cycl/deg)
%              f = Modulus of the spatial frequency (in cycl/deg)
%              g = Overall gain (Recom. value  g=330.74)
%             fm = Parameter that control the exp. decay of the CSFTyler
%                  (Recom. value fm=7.28)      
%              l = Loss at low frequencies (Recom. value fm=0.837)
%              s = Parameter that control the atenuation of the loss factor at 
%                  high frequencies (Recom. value s=1.809)
%              w = Weighting of the Oblique Effect (Recom. value w=1)
%                  (w=0 -> No oblique effect)
%             os = Oblique Effect scale (controls the attenuation of the 
%                  effect at high frequencies).
%                  Recom. value os=6.664).
% 
% This program returns the (spatial domain) FIR filter coefficients to be applied 
% with 'filter2' over the desired image. (These coefficients are similar to the PSF).
%
% Currently the filter design method is frequency sampling (see Image Processing Toolbox
% Tutorial) to (approximately) obtain the desired frequency response, CSFSSO, with the
% required filter order, N (odd), at a particular sampling frequency, fs (in cycles/deg).
%
% USAGE: [h,CSSFO,CSFT,OE]=csfsso(fs,N,g,fm,l,s,w,os);
%
% Recomended Values ( Watson&Ramirez,Modelfest OSA Meeting 1999 )
%
%        [h,CSSFO,CSFT,OE]=csfsso(fs,N,330.74,7.28,0.837,1.809,1,6.664);
%

[fx,fy]=freqspace(N,'meshgrid'); 
fx=fx*fs/2; 
fy=fy*fs/2;  

f=sqrt(fx.^2+fy.^2); 
f=(f>0).*f+0.0001*(f==0);   % To avoid singularity at zero frequency

CSFT=g*(exp(-(f/fm))-l*exp(-(f.^2/s^2)));
OE=1-w*(4*(1-exp(-(f/os))).*fx.^2.*fy.^2)./(f.^4);

CSFSSO=CSFT.*OE;

h=fsamp2(CSFSSO);

