function [xy_buenas,frec_buenas,ind_buenas,colores_map_rgb,er2_t,Mss,Ap,Ap_net,Afit] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2)

%  [xy_buenas,frec_buenas,ind_buenas,colores_map_rgb,error,A,Ap,Ap_net,Afit] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2)

load(file)
% fsx = 25;
% xmax = 4.05;

figure(fig),colormap gray,Ap=disp_patches(Mss,sqrt(size(Mss,2)),b);
% h=figure(fig+1),plot(sort(er2_t)),xlabel('TICA feature'),ylabel('Mean Absolute Error / Range'),set(h,'color',[1 1 1])
% hold on,plot(1:length(er2_t),umbral_error*ones(length(er2_t)),'r-')
% axis([0 size(Mss,2) 0 0.9])

ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);

% umbral_error = 1;

xf_sort=xf2_t;
A_fit = Mss;
AA_nonparam = Mss;

L = size(Mss,2);
N = sqrt(size(Mss,1));

xy_buenas = [];
frec_buenas = [];
ind_buenas = [];

figure(fig+2)
for i=1:L
    x=xf_sort(i,1);
    y=xf_sort(i,2);
    z=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
    [Gs,Gc] = sens_gabor3d_space(N,N,1,fsx,fsx,1,xf_sort(i,1),xf_sort(i,2),0,...
        delt_xf_ang_phase2_t(i,4),xf_sort(i,3),xf_sort(i,4),0,delt_xf_ang_phase2_t(i,1),delt_xf_ang_phase2_t(i,2),delt_xf_ang_phase2_t(i,3),1);    
    if er2_t(i) < umbral_error 
       A_fit(:,i) = Gs(:)';
       AA_nonparam(:,i) = Mss(:,i);
       if (x < x_lim) & (x > 0) & (y < x_lim) & (y > 0)
          plot(x,y,'.','markersize',size_dots_x,'color',map_rgb(z,:)), hold on
          xy_buenas = [xy_buenas;x y];
          ind_buenas = [ind_buenas i];
       end
    else
       A_fit(:,i) = 0*Gs(:)';
       A_fit(1,i) = -1;
       A_fit(end,i) = 1;
       AA_nonparam(:,i) = 0*Gs(:)';
       AA_nonparam(1,i) = -1;
       AA_nonparam(end,i) = 1;
    end
end
axis([-0.05 xmax -0.05 xmax]),axis square,axis ij
xlabel('x (deg)'),ylabel('y (deg)'),
%title('Centers of TICA sensors in the retinal space')
set(gcf,'color',[1 1 1])

figure(fig+100)
for i=1:L
    x=xf_sort(i,1);
    y=xf_sort(i,2);
    z=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
    [Gs,Gc] = sens_gabor3d_space(N,N,1,fsx,fsx,1,xf_sort(i,1),xf_sort(i,2),0,...
        delt_xf_ang_phase2_t(i,4),xf_sort(i,3),xf_sort(i,4),0,delt_xf_ang_phase2_t(i,1),delt_xf_ang_phase2_t(i,2),delt_xf_ang_phase2_t(i,3),1);    
    if er2_t(i) < umbral_error 
       A_fit(:,i) = Gs(:)';
       AA_nonparam(:,i) = Mss(:,i);
       if (x < x_lim) & (x > 0) & (y < x_lim) & (y > 0)
          plot(x,y,'.','markersize',size_dots_x2,'color',map_rgb(z,:)), hold on
          xy_buenas = [xy_buenas;x y];
          ind_buenas = [ind_buenas i];
       end
    else
       A_fit(:,i) = 0*Gs(:)';
       A_fit(1,i) = -1;
       A_fit(end,i) = 1;
       AA_nonparam(:,i) = 0*Gs(:)';
       AA_nonparam(1,i) = -1;
       AA_nonparam(end,i) = 1;
    end
end
xmmm = x_lim;
axis([0 xmmm 0 xmmm]),axis square,axis ij,
% xlabel('x (deg)'),ylabel('y (deg)'),
%title('Centers of TICA sensors in the retinal space')
set(gcf,'color',[1 1 1])
axis off

xf_sort = xf2_t;
figure(fig+3)
for i=1:L
    x=xf_sort(i,1);
    y=xf_sort(i,2);    
    z(i)=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
%     z(i)
%     map_rgb(z(i),:)
    if (er2_t(i) < umbral_error) & (x < x_lim) & (x > 0) & (y < x_lim) & (y > 0)
        plot(xf_sort(i,3),xf_sort(i,4),'.','markersize',size_dots_f,'color',map_rgb(z(i),:)), hold on 
        plot(-xf_sort(i,3),-xf_sort(i,4),'.','markersize',size_dots_f,'color',map_rgb(z(i),:)), hold on 
        frec_buenas = [frec_buenas;xf_sort(i,3),xf_sort(i,4),-xf_sort(i,3),-xf_sort(i,4)];
        axis([-f_lim f_lim -f_lim f_lim])
        %pause(0.1)
    end
end
axis square
%title('Centers of TICA features in the frequency domain')
xlabel('f_x (c/deg)'),ylabel('f_y (c/deg)'),
set(gcf,'color',[1 1 1])

figure(fig+4),colormap gray,Ap_net=disp_patches(AA_nonparam,sqrt(size(Mss,2)),b);
close(fig+4)
figure(fig+5),colormap gray,Afit=disp_patches(A_fit,sqrt(size(Mss,2)),b);
%figure(fig+6),colormap gray,disp_patches(Mss,sqrt(size(Mss,2)),b);

colores_map_rgb = z;