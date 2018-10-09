function x=fft3(x,sif)

%
% FFT3 computes the 3D Fourier transform of a sequence of 2D signals
%      by computing the 1D fft of each (temporal/spectral) line built from
%      the coefficients of the 2D fft of each frame (or spectral band).
%      RESTRICTION: frames (or spectral bands) are assumed to be squared.
%
% SYNTAX: f = fft3( s , fftshift?(0/1));
%
%  s = sequence in 2D array format 
%      Frames/bands stacked side-by-side from left to right 
%      (see now2then for additional comments on sequence formats)
%
%  fftshift = flag to enable (1) or disable (0) the fftshift rearrangement 
%             of the transform coefficients.
%  
%  f = complex coefficients of the Fourier transform in 2D array format
%      

m=size(x);
fotogramas=m(2)/m(1);

if sif==0
   for i=1:fotogramas
          f1=sacafot(x,m(1),m(1),i);
          f1=fft2(f1);
          x=metefot(x,f1,i,1);
   end
   clear f1
   
   a=zeros(1,fotogramas);
   
   for i=1:m(1)
      for j=1:m(1)
   %      a=slineat(x,[i j]);
          for l=1:fotogramas
             a(l)=x(i,(l-1)*m(1)+j);
          end
          a=fft(a);
   %      x=mlineat(x,a,[i j]);
          for l=1:fotogramas
            x(i,(l-1)*m(1)+j)=a(l);
          end
       end
   end
else
   for i=1:fotogramas
          f1=sacafot(x,m(1),m(1),i);
          f1=fftshift(fft2(f1));
          x=metefot(x,f1,i,1);
   end
   clear f1
   a=zeros(1,fotogramas);
   for i=1:m(1)
      for j=1:m(1)
   %      a=slineat(x,[i j]);
          for l=1:fotogramas
             a(l)=x(i,(l-1)*m(1)+j);
          end
          a=fftshift(fft(a));
   %      x=mlineat(x,a,[i j]);
          for l=1:fotogramas
            x(i,(l-1)*m(1)+j)=a(l);
          end
       end
   end
end