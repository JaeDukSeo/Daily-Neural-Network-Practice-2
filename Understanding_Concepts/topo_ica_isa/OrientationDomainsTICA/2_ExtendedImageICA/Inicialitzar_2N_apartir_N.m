% lo que pretenc en estes codi
% apartir de unes funcions bases A dun patch de NorixNori, crear una A de
% 100x100
% pq ho fem
% al llancar el TICA amb imatges de 100x100, algoritme de minimitzacio del 
% error (creiem) que sha aturant a un mimin i ja no convergeixen mes
% funciona base
clear all
addpath(genpath('/media/disk/vista/Papers/PLOS_2016_tica/code'))

path_data='/media/disk/vista/Papers/PLOS_2016_tica/code';
fs=12;
%load([path_data 'tica_2x_50_image_A.mat' ])
load('sortu_2x_32_A.mat')
Nori=32;
Nori2= round(Nori*0.8);
Nnew=2*Nori;
Nnew2= round(Nnew*0.8);

%disp_patches(A,sqrt(size(A,2)),3);

N=size(A,2);
Nori2=sqrt(size(A,2));
Anew=zeros(Nnew^2,Nnew2^2);

%[Ms,xf1,deltas_xf_ang_phase1,xf2,deltas_xf_ang_phase2,xfm,deltas_xf_ang_phasem,err1,err2,errm] = sort_basis(A,fs);

% 1) agafar cada base de A50 i posarla en Anew

 Mj=zeros(Nnew2,Nnew2);
 M2=zeros(Nnew2,Nnew2);
 Mj(1:2:Nnew2,2:2:Nnew2-2)=-1;
 Mj(2:2:Nnew2-2,1:2:Nnew2-1)=-2;
 Mj(2:2:Nnew2-2,2:2:Nnew2-2)=-3;
for i=1:N
    A_aux=reshape(A(:,i),Nori,Nori);

    if mod(i,Nori2)==0
         b=floor(i/Nori2)-1;
    else
        b=floor(i/Nori2);
    end
    c=i-b*Nori2;
    j=Nnew2*(2*b)+2*c-1;
    A_aux=imresize(A_aux,[Nnew,Nnew]);
   
    Anew(:,j)=A_aux(:);
    Mj(2*b+1,2*c-1)=i;
    M2(2*b+1,2*c-1)=2;
end

disp_patches(Anew,Nnew2,3);

% 2) fitt the bases based ont he neigb
% Construir el dominio de fourier y el dominio espacial
columns_x=Nnew;
rows_y=Nnew;
fsx =12.5;  % double frecuencia de mostreo ciclos por grados
fsy=12.5;   % double frecuencia de mostreo ciclos por grados
fst=1;    % doublefrecuencia de mostreo en 1/sec
frames_t = 1;   % int N frames
[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(columns_x,rows_y,frames_t, fsx,fsy,fst);
to=0;     
delta_ft=0;
cont=1;
for i=1:Nnew2
    for j=1:Nnew2
    %-------------------------------------------------
    % FITS
    %--------------------------------------------------
     if  Mj(i,j)==-1 % dreta esquerra
         a=Mj(i,j-1);
         b=Mj(i,j+1);         
     elseif Mj(i,j)==-2%% dalt baix
         a=Mj(i-1,j);
         b=Mj(i+1,j);
     elseif Mj(i,j)==-3 % diagonal
         a=Mj(i-1,j-1);
         b=Mj(i+1,j+1);
     else
         M2(i,j)=1;
         a=[];
     end
     if ~isempty(a)
     ini=0.5*xf2_t(a,:)+0.5*xf2_t(b,:);
     xo=ini(1);
     yo=ini(2);
     fxo=ini(3);
     fyo=ini(4);
     delta_fx=1/2*(delt_xf_ang_phase2_t(a,1)+delt_xf_ang_phase2_t(b,1));
     delta_fy=1/2*(delt_xf_ang_phase2_t(a,2)+delt_xf_ang_phase2_t(b,2)); 
     aux3=1/2*(delt_xf_ang_phase2_t(a,3)+delt_xf_ang_phase2_t(b,3));
     phase=1/2*(delt_xf_ang_phase2_t(a,4)+delt_xf_ang_phase2_t(b,4));
     
     % gabor
     [Gs,Gc] = sens_gabor3d_space(Nori,Nori,1,fsx,fsx,1,xo,yo,0,...
         phase,fxo,fyo,0,delta_fx,delta_fy,aux3,1); 
     % resize
      A_aux=imresize(Gs,[Nnew,Nnew]);
      Anew(:,cont)=A_aux(:);
   
     end
        cont=cont+1;
    end
end

% we have problems if A has Nan or 0 , so put them random.
aux=find(isnan(Anew ));
Anew(aux)=randn(length(aux),1);
aux=find(Anew==0);
Anew(aux)=randn(length(aux),1);
figure
colormap gray
disp_patches(Anew,sqrt(size(Anew,2)),3);

