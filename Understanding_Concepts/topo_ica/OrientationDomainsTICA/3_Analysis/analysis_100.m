load tica_1x_100_image

%
%
% SELECTION OF "FINAL" FEATURES
%
%

indices=reshape(1:6400,[80 80])';

b = 5;
N = 100;
figure,colormap gray,lala=disp_patches(A,80,b);
close all
figure,colormap gray, imagesc(lala),axis square

% First guess
%R1 = [9 52 29 81];
%R2 = [34 40 43 50];
%R3 = [42 4 59 22];
%R4 = [50 30 66 45];

% More accurate guess
R1 = [8 49 29 79];
R2 = [35 40 43 48];
R3 = [41 6 55 21];
R4 = [50 30 67 46];

R = [R1;R2;R3;R4];

% Regions

for i =1:4
    
    fi = (R(i,1)-1)*N + (R(i,1)-1)*b + 1;
    ff = R(i,3)*N + (R(i,3)+1)*b;
    ci = (R(i,2)-1)*N + (R(i,2)-1)*b + 1;
    cf = R(i,4)*N + (R(i,4)+1)*b;
    
    [fi ff ci cf]
    
    region_i = lala( (R(i,1)-1)*N + (R(i,1)-1)*b + 1 : R(i,3)*N + (R(i,3)+1)*b , (R(i,2)-1)*N + (R(i,2)-1)*b + 1 : R(i,4)*N + (R(i,4)+1)*b );
    
    figure(200+i),colormap gray,imagesc(region_i),axis off,axis equal
    
end
    
% Check that the function at location p=(i,j) is what we see in the topology
% p = [60 37];
% ind = indices(p(1),p(2));
% AA = reshape(A(:,ind),[100 100]);
% figure(2),colormap gray,imagesc(AA),axis square

WW = W';
k = 1;
% Funciones medio-buenas (han convergido)
% Funciones criticas: aquellas funciones (en general low freq.) que se fitean con error superior a
% 0.15, pero tienen calidad de fit aceptable. Estas funciones se identificaron a mano 
% 

func_criticas(1).filcol = [8 10;9 9;9 10;9 11;10 8;10 10;10 11;10 12;11 8;11 9;11 11;12 8;12 9;12 10;12 11;13 10;13 11;13 12;9 20;9 21;9 26;7 25;10 26;11 25];
func_criticas(2).filcol = [5 5]; % Por poner la central (en realidad en la region 2 no hay ninguna critica)
func_criticas(3).filcol = [8 6;8 7;8 8;9 8;9 9;11 8];
func_criticas(4).filcol = [7 10;9 7;9 8;9 9;10 8];

fun_critic_en_A = [];
fun_critic_en_regions = [];
posiciones_en_regions = [];

for reg = 1:length(R)
    indis(reg).ind = [];
    pos_crit = func_criticas(reg).filcol;
    for fil = R(reg,1):R(reg,3)
        for col = R(reg,2):R(reg,4)
            ff = fil - R(reg,1) + 1;
            cc = col - R(reg,2) + 1;
            p = [fil col];
            pos = [ff cc];
            ind = indices(p(1),p(2));
            AA = reshape(A(:,ind),[100 100]);
            ww = reshape(WW(:,ind),[100 100]);

            is_crit = sum(prod(pos_crit == repmat(pos,size(pos_crit,1),1),2));
           % figure(2),colormap gray,subplot(121),imagesc(AA),axis square
           % title([num2str(fil),'  ',num2str(col)])
           % figure(2),colormap gray,subplot(122),imagesc(ww),axis square
            indis(reg).ind = [indis(reg).ind ind];
            
            if is_crit ==1
                fun_critic_en_A = [fun_critic_en_A ind];
                fun_critic_en_regions = [fun_critic_en_regions 1];
                posiciones_en_regions = [posiciones_en_regions;pos];
            else
                fun_critic_en_regions = [fun_critic_en_regions 0];
                posiciones_en_regions = [posiciones_en_regions;0 0];
            end
            
            
            k = k+1;
           % pause(0.5)
        end
    end
end
index = [indis(1).ind indis(2).ind indis(3).ind indis(4).ind];

clear W;
AA = A(:,index);
clear A
WWW = WW(:,index);
clear WW

L = length(index);

save analysis_100_b

fs = 25;
[MssA,xf1_tA,delt_xf_ang_phase1_tA,xf2_tA,delt_xf_ang_phase2_tA,xfm_tA,delt_xf_ang_phasem_tA,er1_tA,er2_tA,erm_tA] = sort_basis(AA,fs);
save analysis_100_b
% [MssW,xf1_tW,delt_xf_ang_phase1_tW,xf2_tW,delt_xf_ang_phase2_tW,xfm_tW,delt_xf_ang_phasem_tW,er1_tW,er2_tW,erm_tW] = sort_basis(WWW,fs);
% save analysis_100_b

figure(8),plot(sort(er1_tA),'r-')
figure(8),hold on,plot(sort(er2_tA),'g-')
figure(8),hold on,plot(sort(erm_tA),'b-')
%figure(8),hold on,plot(sort(er1_tW),'r--')
%figure(8),hold on,plot(sort(er2_tW),'g--')
%figure(8),hold on,plot(sort(erm_tW),'b--')

ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);

fsx = 25;

umbral_error = 0.1;

xf_sort=xf2_tA;
xf_sort1=xf1_tA;
A_fit = AA;
A_fit1 = AA;
AA_nonparam = AA;
AA_nonparam1 = AA;

lilo=[index' posiciones_en_regions fun_critic_en_regions'];

figure
for i=1:L
    x=xf_sort(i,1);
    y=xf_sort(i,2);
    z=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
    [Gs,Gc] = sens_gabor3d_space(100,100,1,fsx,fsx,1,xf_sort(i,1),xf_sort(i,2),0,...
        delt_xf_ang_phase2_tA(i,4),xf_sort(i,3),xf_sort(i,4),0,delt_xf_ang_phase2_tA(i,1),delt_xf_ang_phase2_tA(i,2),delt_xf_ang_phase2_tA(i,3),1);    
    if or((er2_tA(i) < umbral_error), fun_critic_en_regions(i)==1) 
       A_fit(:,i) = Gs(:)';
       AA_nonparam(:,i) = AA(:,i);
       plot(x,y,'.','markersize',20,'color',map_rgb(z,:)), hold on 
       if fun_critic_en_regions(i)==1
          lilo(i,:),er2_tA(i)
          figure(100),colormap gray,imagesc(Gs)
          pause
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
axis([-0.05 4.05 -0.05 4.05]),axis square,axis ij
xlabel('x (deg)'),ylabel('y (deg)'),
title('V1 sampling of the spatial domain (TICA) -2-')
set(gcf,'color',[1 1 1])

xf_sort = xf2_tA;
figure
for i=1:L
    z(i)=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
%     z(i)
%     map_rgb(z(i),:)
   
    if or((er2_tA(i) < umbral_error), fun_critic_en_regions(i)==1)
        plot(xf_sort(i,3),xf_sort(i,4),'.','markersize',20,'color',map_rgb(z(i),:)), hold on 
        plot(-xf_sort(i,3),-xf_sort(i,4),'.','markersize',20,'color',map_rgb(z(i),:)), hold on 
        axis([-25/2 25/2 -25/2 25/2])
        % pause(0.1)

    end
end
axis square
title('V1 features in the frequency domain (TICA) -2-')
xlabel('f_x (c/deg)'),ylabel('f_y (c/deg)'),
set(gcf,'color',[1 1 1])

lim = round(sqrt(L));

% figure,colormap gray,imagesc(reshape(er2_tA(1:lim^2),[lim lim])')
% figure,colormap gray,lala=disp_patches(AA_nonparam(:,1:lim^2),lim,b);
% figure,colormap gray,lala=disp_patches(A_fit(:,1:lim^2),lim,b);

% figure,colormap gray,imagesc(reshape(er1_tA(1:lim^2),[lim lim])')
% figure,colormap gray,lala=disp_patches(AA_nonparam1(:,1:lim^2),lim,b);
% figure,colormap gray,lala=disp_patches(A_fit1(:,1:lim^2),lim,b);

% Conclusion de haber mirado 1 2 y 3: el 1 es cutre. 2 y 3 son similares

load tica_1x_100_image A

AAAnp = A;
AAAfit = A;
AAAnp(:,index) = AA_nonparam;
AAAfit(:,index) = A_fit;

b = 5;
N = 100;
figure,colormap gray,lala = disp_patches(A,80,b);
figure,colormap gray,lala_np = disp_patches(AAAnp,80,b);
figure,colormap gray,lala_fit = disp_patches(AAAfit,80,b);

% First guess
%R1 = [9 52 29 81];
%R2 = [34 40 43 50];
%R3 = [42 4 59 22];
%R4 = [50 30 66 45];

% More accurate guess
R1 = [8 49 29 79];
R2 = [35 40 43 48];
R3 = [41 6 55 21];
R4 = [50 30 67 46];

R = [R1;R2;R3;R4];

% Regions

for i =1:4
    
    fi = (R(i,1)-1)*N + (R(i,1)-1)*b + 1;
    ff = R(i,3)*N + (R(i,3)+1)*b;
    ci = (R(i,2)-1)*N + (R(i,2)-1)*b + 1;
    cf = R(i,4)*N + (R(i,4)+1)*b;
    
    [fi ff ci cf]
    
    region_i = lala( (R(i,1)-1)*N + (R(i,1)-1)*b + 1 : R(i,3)*N + (R(i,3)+1)*b , (R(i,2)-1)*N + (R(i,2)-1)*b + 1 : R(i,4)*N + (R(i,4)+1)*b );
    figure(200+i),colormap gray,imagesc(region_i),axis off,axis equal
    
    %region_i = lala_np( (R(i,1)-1)*N + (R(i,1)-1)*b + 1 : R(i,3)*N + (R(i,3)+1)*b , (R(i,2)-1)*N + (R(i,2)-1)*b + 1 : R(i,4)*N + (R(i,4)+1)*b );
    %figure(300+i),colormap gray,imagesc(region_i),axis off,axis equal    
    
    region_i = lala_fit( (R(i,1)-1)*N + (R(i,1)-1)*b + 1 : R(i,3)*N + (R(i,3)+1)*b , (R(i,2)-1)*N + (R(i,2)-1)*b + 1 : R(i,4)*N + (R(i,4)+1)*b );
    figure(400+i),colormap gray,imagesc(region_i),axis off,axis equal    
    
    
end

h = figure,plot(sort(er2_tA)),xlabel('TICA feature'),ylabel('E(error)/E(signal)'),set(h,'color',[1 1 1])
