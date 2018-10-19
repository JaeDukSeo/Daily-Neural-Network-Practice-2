function [fil]=pasaband(fs,N,f1,f2,o,a)

% 'PASABAND' genera un filtro pasa-banda con determinada anchura frecuencial 
% (expresada en ciclos/grado) y angular (orientacion y achura en grados)
%
% USO: filt=pasaband(fs,N,fmin,fmax,orientacion,anchura);
%
% OJO: La anchura no debe ser nula ni exceder de 180 grados
%      La orientacion debe estar entre 0 y 180 grados
 
% kx=linspace(-fs/2,fs/2,N);
% kx=ones(N,1)*kx;
% ky=rot90(kx,3);

[xl,yl,tl,kx,ky,ftl] = spatio_temp_freq_domain(N,N,1,fs,fs,1);

radio2=((kx.^2+ky.^2)<f2^2);
radio1=((kx.^2+ky.^2)<f1^2);
modulo=radio2-radio1;


if o<90
   if o+a/2<90
      c1=(ky<((tan(pi*(o+a/2)/180))*kx));
      c2=(ky>((tan(pi*(o-a/2)/180))*kx));
   elseif o+a/2==90
      c1=(ky>((tan(pi*(o+a/2+1)/180))*kx));
      c2=(ky>((tan(pi*(o-a/2-1)/180))*kx));             
   else
      c1=(ky>((tan(pi*(o+a/2)/180))*kx));
      c2=(ky>((tan(pi*(o-a/2)/180))*kx));
   end
else
   if o-a/2>90
      c1=(ky>((tan(pi*(o+a/2)/180))*kx));
      c2=(ky<((tan(pi*(o-a/2)/180))*kx));
   elseif o-a/2==90
      c1=(ky>((tan(pi*(o+a/2+1)/180))*kx));
      c2=(ky>((tan(pi*(o-a/2-1)/180))*kx));             
   else
      c1=(ky>((tan(pi*(o+a/2)/180))*kx));
      c2=(ky>((tan(pi*(o-a/2)/180))*kx));
   end
end

angul=abs(c2-c1)==0;
fil=angul.*modulo;
fil=fftshift(fil);
