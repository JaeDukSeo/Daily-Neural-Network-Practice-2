function [xf_marina,colores_map_rgb_buenos,error_bueno,error,Afit_manual,Ap]=analysis_100_manual(file,fsx,umbral_error1,func_criticas,fig)


% file = 'sortu_1x_100_1';
% fsx = 25;
xmax = 5.12;
f_lim = fsx/2;
x_lim = 4;
b = 3;
% umbral_error = 0.1;
size_dots_x = 10;
size_dots_f = 15;
%fig=10000;
[xy_buenas,frec_buenas,ind_buenas,colores_map_rgb,error,A,Ap,Ap_net,Afit] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error1,size_dots_x,size_dots_f,fig,2*size_dots_x);

% func_criticas = [8 10;9 9;9 10;9 11;10 8;10 10;10 11;10 12;11 8;11 9;11 11;12 8;12 9;12 10;12 11;13 10;13 11;13 12;9 20;9 21;7 25;9 25;10 25;11 24];
indices=reshape(1:size(A,2),[sqrt(size(A,2)) sqrt(size(A,2))])';
ind_criticos = zeros(1,size(func_criticas,1));
for i=1:size(func_criticas,1)
    ind_criticos(i) = indices(func_criticas(i,1),func_criticas(i,2));
end 
ind_criticos = sort(ind_criticos);

umbral_error2 = 1;
x_lim = 100;
[xy_buenas2,frec_buenas2,ind_buenas2,colores_map_rgb2,error2,A2,Ap2,Ap_net2,Afit2] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error2,size_dots_x,size_dots_f,fig,2*size_dots_x);

Afit_manual = Afit;
N = 100;
for i =1:size(func_criticas,1)
    
    %    fi = (R(i,1)-1)*N + (R(i,1)-1)*b + 1;
    %    ff = R(i,3)*N + (R(i,3)+1)*b;
    %    ci = (R(i,2)-1)*N + (R(i,2)-1)*b + 1;
    %    cf = R(i,4)*N + (R(i,4)+1)*b;

    fi = (func_criticas(i,1)-1)*N + (func_criticas(i,1)-1)*b + 1;
    ff = func_criticas(i,1)*N + (func_criticas(i,1)+1)*b;
    ci = (func_criticas(i,2)-1)*N + (func_criticas(i,2)-1)*b + 1;
    cf = func_criticas(i,2)*N + (func_criticas(i,2)+1)*b;
    
   % [i fi ff ci cf]
    
    Afit_manual( fi : ff , ci : cf ) = Afit2( fi : ff , ci : cf );
    
    %     if i>21
    %        figure(1000),imagesc(Afit_manual( fi : ff , ci-100-b : cf ))
    %        [fi ff ci cf]
    %        [min(min(Afit_manual( fi : ff , ci : cf ))) max(max(Afit_manual( fi : ff , ci : cf )))]
    %        pause
    %     end
end

close(fig+0),close(fig+2),close(fig+3),close(fig+5),%close(fig+2),close(fig+3), close(fig+6),close(fig+7),close(fig+8),close(fig+9),close(fig+10),close(fig+11),
% figure(fig),colormap gray,imagesc(Afit_manual(1:2166,:)),axis off,axis equal
% figure(fig+1),colormap gray,imagesc(Ap(1:2166,:)),axis off,axis equal

xy_buenas_buenas = xy_buenas;
frec_buenas_buenas = frec_buenas;
ind_buenas_buenas =ind_buenas;
colores_map_rgb_buenos = colores_map_rgb(ind_buenas);
error_bueno = error(ind_buenas);

k = 1;
for i = 1:length(ind_criticos)
    for j=1:length(ind_buenas2)
        % [i j]
        if sum(ind_buenas2(j) == ind_criticos(i))>0
           xy_buenas_buenas = [xy_buenas_buenas;xy_buenas2(j,:)];
           frec_buenas_buenas = [frec_buenas_buenas;frec_buenas2(j,:)];
           ind_buenas_buenas = [ind_buenas_buenas ind_buenas2(j)];
           colores_map_rgb_buenos = [colores_map_rgb_buenos colores_map_rgb2(j)];
           error_bueno = [error_bueno error2(j)];
           %k
           %k=k+1;
        end
    end
end

xf_marina = [xy_buenas_buenas frec_buenas_buenas(:,1:2)];

ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);

figure(fig+2)
xf_sort = xy_buenas_buenas;
for i=1:size(xf_sort,1)
    x=xf_sort(i,1);
    y=xf_sort(i,2);
    z=colores_map_rgb_buenos(i);
       if (x < x_lim) & (y < x_lim)
          plot(x,y,'.','markersize',size_dots_x,'color',map_rgb(z,:)), hold on
       end
end
axis([-0.05 xmax -0.05 xmax]),axis square,axis ij
xlabel('x (deg)'),ylabel('y (deg)'),
%title('Centers of TICA sensors in the retinal space')
set(gcf,'color',[1 1 1])

xf_sort = frec_buenas_buenas;
figure(fig+3)
for i=1:size(xf_sort,1)
    z=colores_map_rgb_buenos(i);
        plot(xf_sort(i,3),xf_sort(i,4),'.','markersize',size_dots_f,'color',map_rgb(z,:)), hold on 
        plot(-xf_sort(i,3),-xf_sort(i,4),'.','markersize',size_dots_f,'color',map_rgb(z,:)), hold on 
end
axis([-f_lim f_lim -f_lim f_lim])
axis square
%title('Centers of TICA features in the frequency domain')
xlabel('f_x (c/deg)'),ylabel('f_y (c/deg)'),
set(gcf,'color',[1 1 1])
