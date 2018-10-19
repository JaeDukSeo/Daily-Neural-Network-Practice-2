addpath(genpath('/media/disk/vista/Papers/IMAGING/intrinsic_Imaging'))

load F2141_Full__E7B0.BLK.mat

angle = [0 0;22.5 202.5; 112.5 292.5;45 225;135 315; 67.5 247.5;157 337; 90 -90;180 360]

m=min([c0(:);c1(:);c2(:);c3(:);c4(:);c5(:);c6(:);c6(:);c7(:);c8(:)]);
M=max([c0(:);c1(:);c2(:);c3(:);c4(:);c5(:);c6(:);c6(:);c7(:);c8(:)]);

cc0 = (c0-m)/(M-m);
cc1 = (c1-m)/(M-m);
cc2 = (c2-m)/(M-m);
cc3 = (c3-m)/(M-m);
cc4 = (c4-m)/(M-m);
cc5 = (c5-m)/(M-m);
cc6 = (c6-m)/(M-m);
cc8 = (c8-m)/(M-m);

cc11 = ((cc1-cc0)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
cc22 = ((cc2-cc0)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));
ccR = ((cc1-cc2)./(cc0+cc1+cc2+cc3+cc4+cc5+cc6+cc7+cc8));

cc2n = 

std_n = std(cc2n(:)); 
mm = mean(cc2n(:)) - std_n;
MM = mean(cc2n(:)) + std_n;

cc2n = (cc2n - mm) / (MM-mm);

for i=1:20
figure(1),colormap gray,imagesc(cc2n(:,:,i),[0 1])
pause
i
end
