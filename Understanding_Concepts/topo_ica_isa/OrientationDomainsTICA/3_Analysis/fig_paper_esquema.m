clear all
%close all

Vfs(1:5)=25;
Vfs([6 8])=12.5;
Vfs([7 9])=6.25;
Fonttitle=16;
SizeDots=20;
Nrandom=10000;
Nclas=4;
Ngrau=180/Nclas;
GrauIn=0;
bordes=0.1;
%load(['KL_2x_32_A_Nclas' num2str(Nclas) '_Nrand_' num2str(Nrandom) '_ori' num2str(floor(GrauIn)) ],'Dist_Test')
 
load(['sortu_2x_32_A.mat'])
fs=Vfs(6); 

    [nfA,lala]=size(Mss);
    N=sqrt(nfA); 
    xf_sort = xf2_t;
    
    figure('units','normalized','outerposition',[0 0 1 1])
    % xf_sort=xf2_t;
    z=round(359*(pi+atan2(xf_sort(:,4),xf_sort(:,3)))/(2*pi))+1;   
    x=xf_sort(:,1);
    y=xf_sort(:,2);
    c=N/fs*bordes;
    xmin=c;
    x_lim= N/fs-c;
    % square atround the retinal field   
    InSquare=(x>c & x<=x_lim  & y<=x_lim & y>c);
    x=x(InSquare);
    y=y(InSquare );
    z=z(InSquare );    
    Nsen=length(y); % sensors in the scuare 
    %-----------------------------------
    % Discretizar el continuo de angulos en 6 (Nclas)
    Graus=Ngrau/2;    
    VGRaus=GrauIn:Ngrau:180-Graus;
    z(z>180)=z(z>180)-180; 
    z2=zeros(size(z));    
    VGRausS=VGRaus+Graus;
    VGRausI=VGRaus-Graus;   
    for i= 1:length(VGRaus)
        if i==1
            z2(0<z & z<=VGRausS(i))=i;
            z2(VGRausS(end)<z & z<=180)=i;            
        else
            z2(VGRausI(i)<z & z<=VGRausS(i))=i;
        end
    end 
% generados los datos vamos a las simulaciones
i=4
NsenSq1=ceil(sqrt(sum(z2==i))); 
Nsen=sum(z2==i);

%5Vindex=xmin:(x_lim-xmin)/(NsenSq1):x_lim;
Vindex=linspace(xmin,x_lim,NsenSq1);
bins{2}=linspace(c,x_lim,NsenSq1);
bins{1}=linspace(c,x_lim,NsenSq1);

subplot(2,3,1)%-----------------------------------------------
plot(x(z2==i),y(z2==i),'.b','markersize', SizeDots),axis square,axis ij, hold on
%ylabel(['Degrees'])
title([ 'S_{ori}'  ], 'Fontsize', Fonttitle)  
%xlabel(['Degrees'])

subplot(2,3,4)%------------------------------------------------------------
[Ps1,bn]=hist3([x(z2==i),y(z2==i)],'Edges' ,bins); 
pcolor(linspace(xmin,x_lim,NsenSq1),linspace(xmin,x_lim,NsenSq1),Ps1') ...
    ,axis square,axis ij
%-------------------------------------------------------------------
% part random de la figura

subplot(2,3,2)%---------------------------------------------------------
xr=rand(Nsen,1)*(N/fs-2*c)+c;
yr=rand(Nsen,1)*(N/fs-2*c)+c;  
plot(xr,yr,'.b','markersize', SizeDots),axis square , axis ij 
title(['S_{rand} ' ], 'Fontsize', Fonttitle)
%ylabel(['Degrees'])

subplot(2,3,5)%---------------------------------------------------------
[Ps1,bn]=hist3([xr, yr],'Edges' ,bins);
P=Ps1;
pcolor(linspace(xmin,x_lim,NsenSq1),linspace(xmin,x_lim,NsenSq1),P') ,axis square, axis ij

subplot(2,3,3)%---------------------------------------------------------
[xS,yS]=meshgrid(bins{2},bins{2});
xS=xS(:);
yS=yS(:);
aux=(x_lim-xmin)/NsenSq1;
aux=aux/2;
plot(xS+aux,yS+aux,'.b','markersize', SizeDots),axis square, axis ij
axis([xmin x_lim xmin x_lim]) 
title(['S_{cart}' ], 'Fontsize', Fonttitle)
%ylabel(['Degrees'])
%xlabel(['Degrees'])

subplot(2,3,6)%---------------------------------------------------------
[Q,bn]=hist3([xS,yS],'Edges',bins);
pcolor(linspace(xmin,x_lim,NsenSq1),linspace(xmin,x_lim,NsenSq1),Q'),axis square, axis ij

%arreglar les figures ----------------------------------------------- 
for i=4:6
    subplot(2,3,i)
    colormap gray
    caxis([0  max(P(:))])
    set(gca,'Xtick', [],'XtickLabel',[],'Ytick',...
    [],'YtickLabel',[])%$, 'yaxislocation','right')
end
for i=1:3
    subplot(2,3,i)
    set(gca,'Xtick', Vindex,'XtickLabel',[],'Ytick', Vindex,'YtickLabel',[])
    grid on
    axis([xmin x_lim xmin x_lim]) 
    
end
  save2pdf('Test')
%---------------------------------------
% i=3   
% Mean_Dist= mean(Dist_Test(2:end,:));
%  Std_Dist= std(Dist_Test(2:end,:));
% subplot(2,3,[8,9])
% [a,b]=hist(Dist_Test(2:end,i),round(sqrt(Nrandom)/2));
% stairs(b,a/sum(a)), hold on
% plot(Dist_Test(1,i),0,'xr')
% xmean=Mean_Dist(i)-2.576*Std_Dist(i)/sqrt(Nrandom);
% ymean=Mean_Dist(i)+2.576*Std_Dist(i)/sqrt(Nrandom);
% plot([xmean xmean],[0 max(a/sum(a)) ],'b')
% plot([ymean ymean],[0 max(a/sum(a)) ],'b')
% xlabel(['Pr(KL(S_{rand},S_{Cart}))'])  