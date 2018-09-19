addpath(genpath('/media/disk/vista/Papers/IMAGING/intrinsic_Imaging'))

load F2141_Full__E7B0.BLK.mat

angle = [0 0;22.5 202.5; 112.5 292.5;45 225;135 315; 67.5 247.5;157 337; 90 -90;180 360]

% Ortogonales
%    c0 no stimul
%    c1 - c2
%    c3 - c4
%    c5 - c6
%    c7 - c8
%

m = min([c0(:);c1(:);c2(:);c3(:);c4(:);c5(:);c6(:);c6(:);c7(:);c8(:)]);
M = max([c0(:);c1(:);c2(:);c3(:);c4(:);c5(:);c6(:);c6(:);c7(:);c8(:)]);

cc0 = (c0-m)/(M-m);
cc1 = (c1-m)/(M-m);
cc2 = (c2-m)/(M-m);
cc3 = (c3-m)/(M-m);
cc4 = (c4-m)/(M-m);
cc5 = (c5-m)/(M-m);
cc6 = (c6-m)/(M-m);
cc7 = (c7-m)/(M-m);
cc8 = (c8-m)/(M-m);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc11 = ((cc1)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
cc22 = ((cc2)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
ccR = ((cc1-cc2)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));

c11 = norm_image(cc11);
c22 = norm_image(cc22);
ccR = norm_image(ccR);

% c1,c2 - 10 c5,c6 - 18, c7,c8 - 12

ext_row = [600 850];
ext_col = [100 350];
i = 8:12
    figure(1),subplot(121),colormap gray,imagesc(mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(2,1)))
              subplot(122),colormap gray,imagesc(mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(3,2)))
             % subplot(133),colormap gray,imagesc(mean(ccR(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])
     
ccc111 = mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
ccc222 = mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
             
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% cc11 = ((cc3)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
% cc22 = ((cc4)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
% ccR = ((cc3-cc4)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
% 
% cc33 = norm_image(cc11);
% cc44 = norm_image(cc22);
% ccR = norm_image(ccR);
% 
% % c1,c2 - 10 c5,c6 - 18, c7,c8 - 12
% 
% ext_row = [600 850];
% ext_col = [100 350];
% for i = 1:20
%     figure(4),subplot(121),colormap gray,imagesc(mean(cc33(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])
%               subplot(122),colormap gray,imagesc(mean(cc44(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])
%              % subplot(133),colormap gray,imagesc(mean(ccR(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])             
%              pause
%              i
% end 
             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%             

cc11 = ((cc5)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
cc22 = ((cc6)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
ccR = ((cc5-cc6)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));

cc55 = norm_image(cc11);
cc66 = norm_image(cc22);
ccR = norm_image(ccR);

% c1,c2 - 10 c5,c6 - 18, c7,c8 - 12

i = 16:20
    figure(2),subplot(121),colormap gray,imagesc(mean(cc55(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(6,1)))
              subplot(122),colormap gray,imagesc(mean(cc66(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(7,2)))
             % subplot(133),colormap gray,imagesc(mean(ccR(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])

ccc555 = mean(cc55(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
ccc666 = mean(cc66(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc11 = ((cc7)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
cc22 = ((cc8)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
ccR = ((cc7-cc8)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));

cc77 = norm_image(cc11);
cc88 = norm_image(cc22);
ccR = norm_image(ccR);

% c1,c2 - 10 c5,c6 - 18, c7,c8 - 12

i = 10:14
    figure(3),subplot(121),colormap gray,imagesc(mean(cc77(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(8,1)))
              subplot(122),colormap gray,imagesc(mean(cc88(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(9,1)))
             % subplot(133),colormap gray,imagesc(mean(ccR(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])


ccc777 = mean(cc77(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
ccc888 = mean(cc88(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CURVA DE SINTONIZADO 

ang = [292.5 67.5 337 90 180];
res = [cc22(175,185) cc55(175,185) cc66(175,185) cc77(175,185) cc88(175,185)];

[a,ind] = sort(ang);

figure,plot([ang(end)-360 ang(ind(1:end-1))],[res(end) res(ind(1:end-1))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A=ccc111(10:80,70:240);figure,colormap gray,imagesc(  (ccc111 < median(A(:))).*(1-ccc111) )
A=ccc222(10:80,70:240);figure,colormap gray,imagesc(  (ccc222 < median(A(:))).*(1-ccc222) )
A=ccc555(10:80,70:240);figure,colormap gray,imagesc(  (ccc555 < median(A(:))).*(1-ccc555) )
A=ccc666(10:80,70:240);figure,colormap gray,imagesc(  (ccc666 < median(A(:))).*(1-ccc666) )

fact_median = 0.85;

A=ccc111(10:80,70:240);figure,colormap gray,imagesc(  (ccc111 < median(A(:))) );p22 = ccc111 < fact_median*median(A(:));
A=ccc222(10:80,70:240);figure,colormap gray,imagesc(  (ccc222 < median(A(:))) );p67 = ccc222 < fact_median*median(A(:));
A=ccc555(10:80,70:240);figure,colormap gray,imagesc(  (ccc555 < median(A(:))) );p45 = ccc555 < fact_median*median(A(:));
A=ccc666(10:80,70:240);figure,colormap gray,imagesc(  (ccc666 < median(A(:))) );p157= ccc666 < fact_median*median(A(:));

% rojo amarillo verde azul

fact = 0.5;

P22(:,:,1) = fact*p22;
P22(:,:,2) = 0*p22;
P22(:,:,3) = 0*p22;

P67(:,:,1) = fact*p67;
P67(:,:,2) = fact*p67;
P67(:,:,3) = 0*p67;

P45(:,:,1) = 0*p45;
P45(:,:,2) = fact*p45;
P45(:,:,3) = 0*p45;

P157(:,:,1) = 0*p157;
P157(:,:,2) = 0*p157;
P157(:,:,3) = fact*p157;

figure,imagesc(P22)
figure,imagesc(P67)
figure,imagesc(P45)
figure,imagesc(P157)

imm = P22+P67+P45+P157;

figure,imagesc(imm)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dif1 = ccc111 - ccc222;
dif2 = ccc555 - ccc666;

figure,colormap gray,imagesc(dif1)
figure,colormap gray,imagesc(dif2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear P*
load('/media/disk/vista/Papers/IMAGING/intrinsic_Imaging/Ferret/3824/ferret 3824/F_7024__E4B0.BLK.mat')

m = min([c0(:);c1(:);c2(:);c3(:);c4(:)]);
M = max([c0(:);c1(:);c2(:);c3(:);c4(:)]);

cc0 = (c0-m)/(M-m);
cc1 = (c1-m)/(M-m);
cc2 = (c2-m)/(M-m);
cc3 = (c3-m)/(M-m);
cc4 = (c4-m)/(M-m);

angle = [0 0;0 180;90 270;45 225;135 315];

cc11 = ((cc1)./(cc0+cc1+cc2+cc3+cc4));
cc22 = ((cc2)./(cc0+cc1+cc2+cc3+cc4));
ccR = ((cc1-cc2)./(cc0+cc1+cc2+cc3+cc4));

c11 = norm_image(cc11);
c22 = norm_image(cc22);
ccR = norm_image(ccR);

% c1,c2 - 7 c3,c4 - 8,

%ext_row = [600 850];
%ext_col = [100 350];

ext_row = [100 824];
ext_col = [40 300];

i = 5:9
    figure(1),subplot(121),colormap gray,imagesc(mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(2,1)))
              subplot(122),colormap gray,imagesc(mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(3,2)))
             % subplot(133),colormap gray,imagesc(mean(ccR(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])


ccc111 = mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
ccc222 = mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);

%%%%%%%%%%%%%%%%%%%%%

cc11 = ((cc3)./(cc0+cc1+cc2+cc3+cc4));
cc22 = ((cc4)./(cc0+cc1+cc2+cc3+cc4));
ccR = ((cc1-cc2)./(cc0+cc1+cc2+cc3+cc4));

c11 = norm_image(cc11);
c22 = norm_image(cc22);
ccR = norm_image(ccR);

% c1,c2 - 7 c3,c4 - 8,

%ext_row = [600 850];
%ext_col = [100 350];

ext_row = [100 824];
ext_col = [40 300];

i = 6:10
    figure(1),subplot(121),colormap gray,imagesc(mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(2,1)))
              subplot(122),colormap gray,imagesc(mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),title(num2str(angle(3,2)))
             % subplot(133),colormap gray,imagesc(mean(ccR(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1])


ccc333 = mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);
ccc444 = mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3);

%%%%%%%%%%%%%%%%%%%%%

fact_median = 0.8;

A=ccc111(145:600,55:210);figure,colormap gray,imagesc( A );figure,colormap gray,imagesc(  (ccc111 < median(A(:))) );p22 = ccc111 < fact_median*median(A(:));
A=ccc222(145:600,55:210);figure,colormap gray,imagesc( A );figure,colormap gray,imagesc(  (ccc222 < median(A(:))) );p67 = ccc222 < fact_median*median(A(:));
A=ccc333(145:600,55:210);figure,colormap gray,imagesc( A );figure,colormap gray,imagesc(  (ccc333 < median(A(:))) );p45 = ccc333 < fact_median*median(A(:));
A=ccc444(145:600,55:210);figure,colormap gray,imagesc( A );figure,colormap gray,imagesc(  (ccc444 < median(A(:))) );p157= ccc444 < fact_median*median(A(:));

% rojo amarillo verde azul

fact = 0.5;

P22(:,:,1) = fact*p22;
P22(:,:,2) = 0*p22;
P22(:,:,3) = 0*p22;

P67(:,:,1) = fact*p67;
P67(:,:,2) = fact*p67;
P67(:,:,3) = 0*p67;

P45(:,:,1) = 0*p45;
P45(:,:,2) = fact*p45;
P45(:,:,3) = 0*p45;

P157(:,:,1) = 0*p157;
P157(:,:,2) = 0*p157;
P157(:,:,3) = fact*p157;

figure,imagesc(P22)
figure,imagesc(P67)
figure,imagesc(P45)
figure,imagesc(P157)

imm = P22+P67+P45+P157;

figure,imagesc(imm)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('/media/disk/vista/Papers/IMAGING/intrinsic_Imaging/Ferret/3824/ferret 3824/3824_RET__E5B0.BLK.mat')

m = min([c0(:);c1(:);c2(:);c3(:)]);
M = max([c0(:);c1(:);c2(:);c3(:)]);

cc0 = (c0-m)/(M-m);
cc1 = (c1-m)/(M-m);
cc2 = (c2-m)/(M-m);
cc3 = (c3-m)/(M-m);

cc11 = ((cc1)./(cc0+cc1+cc2+cc3));
cc22 = ((cc2)./(cc0+cc1+cc2+cc3));
cc33 = ((cc3)./(cc0+cc1+cc2+cc3));

c11 = norm_image(cc11);
c22 = norm_image(cc22);
c33 = norm_image(cc33);

% c1,c2 - 7 c3,c4 - 8,

%ext_row = [600 850];
%ext_col = [100 350];

ext_row = [1 924];
ext_col = [1 492];

for i = 1:20
    figure(1),subplot(131),colormap gray,imagesc(mean(c11(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),
              subplot(132),colormap gray,imagesc(mean(c22(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),
              subplot(133),colormap gray,imagesc(mean(c33(ext_row(1):ext_row(2),ext_col(1):ext_col(2),i),3),[0 1]),
              i
              pause     
end
