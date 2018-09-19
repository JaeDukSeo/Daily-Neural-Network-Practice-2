function [csfrg,csfyb]=csf_chrom(N,fs);

% CSF_CHROM computes the spatial CSFs for the chromatic channels RG and YB
% approximately reproducing the data in K. Mullen 85 Vis. Res. paper.
%
%  fs = sampling frequency (in cl/deg).
%  N = size of the square discrete domain (in pixels).
% 
% [csf_rg,csf_yb]=csf_chrom(N,fs);

% CSF en el dominio de Fourier

[F1,F2] = freqspace([N N],'meshgrid');
F1=F1*fs/2;
F2=F2*fs/2;

F=sqrt(F1.^2+F2.^2);

csfrg=zeros(N,N);
csfyb=zeros(N,N);
for i=1:N
        [iaf_rg,csf_c]=iafrg(F(i,:),0.1,1,[0 0 0]);
        csfrg(i,:)=csf_c;        
        [iaf_yb,csf_c]=iafyb(F(i,:),0.1,1,[0 0 0]);
        csfyb(i,:)=csf_c;        
end

% Factores comprobados para escalar las CSFs segun las proporciones de Mullen 
% (me pase alguna tarde con esto al tratar de sacar los parametros para el modelo de la DCT cuando la patente)
% El factor global 201.3 es el maximo de la CSF acromatica del SSO

fact_rg=0.75;
fact_yb=0.55; 
max_CSF_achro=201.3;

csfrg=fact_rg*max_CSF_achro*csfrg/max(max(csfrg));
csfyb=fact_yb*max_CSF_achro*csfyb/max(max(csfyb));