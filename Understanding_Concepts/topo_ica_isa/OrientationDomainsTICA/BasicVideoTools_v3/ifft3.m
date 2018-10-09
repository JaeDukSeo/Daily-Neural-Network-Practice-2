function x=ifft3(x,esif)

%
% IFFT3 computes the inverse 3D Fourier transform 
%      It computes the ifft of each (temporal/spectral) frequency line 
%      and then computes the ifft2 of the resulting frames.
%      RESTRICTION: frames (or spectral bands) are assumed to be squared.
%
% SYNTAX: s = ifft3( f , fftshift?(0/1));
%
%  f = complex coefficients of the 3D Fourier transform in 2D array format:
%      Frames of fixed temporal frequency stacked side-by-side from left to right 
%      (see now2then for additional comments on sequence formats)
%
%  fftshift = flag describing whether (1) the Fourier data have been shifted and the
%             origin is in the center, or (0) the discrete transform has not been shifted
%             and hence thre origin is in the top-left corner.
%  
%  s = reconstructed sequence in the spatio-temporal domain (2D array format).
%      Note that the sequence may be complex if the Fourier transform of
%      real data has been manipulated.
%      


m=size(x);
fotogramas=m(2)/m(1);

a=zeros(1,fotogramas);

if esif==0
   for i=1:m(1)
      for j=1:m(1)
         for l=1:fotogramas
             a(l)=x(i,(l-1)*m(1)+j);
         end
         a=ifft(a);
         for l=1:fotogramas
             x(i,(l-1)*m(1)+j)=a(l);
         end
      end
   end

   for i=1:fotogramas
          f1=sacafot(x,m(1),m(1),i);
          f1=ifft2(f1);
          x=metefot(x,f1,i,1);
   end
   clear f1
else
   for i=1:m(1)
      for j=1:m(1)
         for l=1:fotogramas
             a(l)=x(i,(l-1)*m(1)+j);
         end
         a=ifft(ifftshift(a));
         for l=1:fotogramas
             x(i,(l-1)*m(1)+j)=a(l);
         end
      end
   end
   for i=1:fotogramas
          f1=sacafot(x,m(1),m(1),i);
          f1=ifft2(ifftshift(f1));
          x=metefot(x,f1,i,1);
   end
   clear f1
end