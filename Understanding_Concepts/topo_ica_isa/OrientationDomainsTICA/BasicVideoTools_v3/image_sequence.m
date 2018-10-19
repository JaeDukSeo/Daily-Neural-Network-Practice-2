function [s,vr]=image_sequence(im,centro,ne,nt,fse,fst,v,interpol)

%
% IMAGE_SEQUENCE generates a movie sequence of controlled speed from a still image.
% The output 3D spatio-temporal sequence is given in 2D array format (see
% now2then for additional details on the format).
% Sampling frequencies, size of the sequence, and speed are controlled.
%
% The size of the signal domain is defined by:
%      nx = number of rows and columns of each frame (we assume squared frames)
%      nf = number of frames
% Sampling frequencies are controlled by:
%      fsx = spatial sampling frequency (in cycl/deg)
%      fst = temporal sampling frequency (in Hz)
% Speed:
%      v = [vy vx] in deg/sec 
%          Note that vertical speed, vy, comes first because it affects rows
%          while horizontal speed, vx, affects columns.
% The sequence starts from a given location (row,colum) of the image.
%
% The program requires a single still image (big image of size much bigger than nx)
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
% SYNTAX: [ s , v_true ] = image_sequence( im , [init_row init_col] , nx , nf , fsx , fst, [vy vx], interpol)
% 

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

% 
% % s1=rand(round(l1),round(l2));
% 
% [ss,fil,s1]=rcolor2d(fse,l,fm,fM,orient,anch_ang,1,1);
% clear ss fil
% 
% s1=(s1-mini(s1))/(maxi(s1)-mini(s1));

s1=im;

for i=1:nt
    if i==1
       s=sacasub(s1,floor(centro),[ne ne],0);
    else
       s=[s sacasub(s1,floor(centro)+D*(i-1)*fse,[ne ne],interpol)];
    end
end