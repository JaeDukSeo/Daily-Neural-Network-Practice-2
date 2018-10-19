function [s,vr] = noise_sequence(ne,nt,fse,fst,fm,fM,anch_ang,orient,v,interpol)

% NOISE_SEQUENCE generates a discrete sequence of colored noise that moves at certain uniform speed 
% Sampling frequencies, size of the sequence, spatial frequency of colored noise and speed are
% controlled.
%
% The size of the signal domain is defined by:
%      nx = number of rows and columns of each frame (we assume squared frames)
%      nf = number of frames
% Sampling frequencies are controlled by:
%      fsx = spatial sampling frequency (in cycl/deg)
%      fst = temporal sampling frequency (in Hz)
% Colored noise (circular sectors) depends on:
%      . Minimum and maximum spatial frequency: fm, fM (in cycl/deg)
%      . Orientation. orient (in degrees)
%      . Orientation width: angul_wdth (in degrees).
% Speed:
%      v = [vy vx] in deg/sec 
%          Note that vertical speed, vy, comes first because it affects rows
%          while horizontal speed, vx, affects columns.
%
% The program computes a single noisy image (big image of size much bigger than nx)
% and extracts submatrices of size nx*nx with displacements of D = v/fst
% pixels per frame. If this quotient leads to non-integer displacements there are 
% two possibilities:
% 
%  - Interpolation (interpol=1)   -> the desired speed is preserved but luminance errors
%                        will be introduced because of the (bilinear) interpolation.
%                        This will have an effect on the spatial spectrum. 
%
%  - No interpolation (interpol=0) -> the speed will be rounded to lead to integer-pixel displacements
%                        the luminance and spatial spectrum will be preserved but the orientation
%                        of the 3D spectrum (or speed) will be distorted (quantized). 
% 
% The program returns the spatio-temporal sequence (in 2D array format) and
% the actual speed (that will be close to, but different from, the required speed if no
% interpolation is selected).
%
% SYNTAX: [ s, v_true] = noise_sequence(nx,nf,fsx,fst,fm,fM,angul_wdth,orient,[vy vx],interpol(0/1)?);
%

if anch_ang>179.5
   anch_ang = 179.5; 
end

if interpol==0
   D=round(v/fst);
   Dm=D*(nt-1);
   vr=D*fst;
else
   D=v/fst;
   Dm=D*(nt-1);
   vr=D*fst;
end

l1=ne+2*abs(Dm(1))*fse+1;
l2=ne+2*abs(Dm(2))*fse+1;

l=round(max([l1 l2]));

% s1=rand(round(l1),round(l2));

[ss,fil,s1]=rcolor2d(fse,l,fm,fM,orient,anch_ang,1,1);
clear ss fil

s1=(s1-mini(s1))/(maxi(s1)-mini(s1));

for i=1:nt
    if i==1
       s=sacasub(s1,floor([l1/2 l2/2]),[ne ne],0);
    else
       s=[s sacasub(s1,floor([l1/2 l2/2])+D*(i-1)*fse,[ne ne],interpol)];
    end
end