% load('imagen_guay.mat')
nameEst{1}='F_7024__E11B0.BLK.mat';
nameEst{2}='F_7024__E10B0.BLK.mat';
nameEst{4}='F_7024__E4B0.BLK.mat';
load(['/media/disk/vista/Papers/IMAGING/intrinsic_Imaging/Ferret/3824/ferret 3824/'  nameEst{4}])

Vmin = min([c0(:);c1(:);c2(:);c3(:);c4(:)]);
M = max([c0(:);c1(:);c2(:);c3(:);c4(:)]);

cc0 = (c0-Vmin)/(M-Vmin);
cc1 = (c1-Vmin)/(M-Vmin);
cc2 = (c2-Vmin)/(M-Vmin);
cc3 = (c3-Vmin)/(M-Vmin);
cc4 = (c4-Vmin)/(M-Vmin);

angle = [0 0;0 180;90 270;45 225;135 315];
Sumtot=cc0+cc1+cc2+cc3+cc4;
cc11 = cc1./Sumtot;
cc22 = cc2./Sumtot;

c11 = norm_image(cc11);
c22 = norm_image(cc22);


imC22=CrossCorrImage_MMG_v2(c22,1,Index_CC);
im=imC22;
%im=singcockt2;
im=(im-min(im(:)))/(max(im(:))-min(im(:)));
im(im>mean(im(:))+3*std(im(:)))=mean(im(:))+3*std(im(:));
xmin= mean(im(:))-std(im(:));
xmax= mean(im(:))+0*std(im(:));



outp=marina_colum3(im,params);
figure
%imagesc(outp.ROI_map)
subplot(2,1,1)
imagesc(mean(c22(:,:,5:9),3),[xmin xmax] ), colormap gray,  axis equal
subplot(2,1,2)
imagesc(im ), colormap gray,  axis equal

for i=1:outp.N_ROI
   hold on
   bound=outp.Boundery{i};
   plot(bound(:,2),bound(:,1),'b')
end