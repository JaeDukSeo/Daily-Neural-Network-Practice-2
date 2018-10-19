clear all
%close all
Vnom=cell(7,1);
Vnom{1}= 'sortu_1x_100_.mat';  
Vnom{2}= 'sortu_1x_16.mat';% A 256x 169   13
Vnom{3}= 'sortu_1x_20.mat';% A 400x 256    16
Vnom{4}= 'sortu_1x_32.mat';% A 1024 x 676    676 = 26x26  
Vnom{5}= 'sortu_1x_50.mat';% A 2500 x 676    1600 = 40x40
% fs= 12
Vnom{6}= 'sortu_2x_32_A.mat';% A 1024 x 676    676 = 26x26
Vnom{8}= 'sortu_2x_50_A.mat';%
% fs 6
Vnom{7}= 'sortu_4x_32_A.mat';%   
Vnom{9}= 'sortu_4x_50_A.mat';%
Vfs(1:5)=25;
Vfs([6 8])=12.5;
Vfs([7 9])=6.25;
Nrandom=10000;
bordes=0.1;
GrauIn=0;% graus inicial del quesito
cont=1;
for Nclas=[4 5 6];
    Ngrau=180/Nclas;
for jj=2:9
    fs=Vfs(jj); 
    load([Vnom{jj}  ])
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
   
    load(['KL_' Vnom{jj}(7:end-4) '_Nclas' num2str(Nclas) '_Nrand_' num2str(Nrandom) '_ori' num2str(floor(GrauIn)) ],'Dist_Test')
    %---------------------------------------------------------------------
    % mean int
    Mean_Dist= mean(Dist_Test(2:end,:));
    Std_Dist= std(Dist_Test(2:end,:));
    
    for i=1:Nclas
        pval=sum(Dist_Test(1,i)>Dist_Test(2:end,i))/Nrandom;      
       
        NsenSq1=ceil(sqrt(sum(z2==i))); 
        NsenSq2=floor(sqrt(sum(z2==i)));
        subplot(1,Nclas,i)
%         Vindex=xmin:(x_lim-xmin)/(NsenSq1):x_lim;
%         plot(x(z2==i),y(z2==i),'.b','markersize',20),axis square,axis ij, hold on
%         set(gca,'Xtick', Vindex,'XtickLabel',[],'Ytick', Vindex,'YtickLabel',[])
%         grid on   
%         title([ 'S_{ori} for ' num2str(VGRaus(i)) ' deg'  ]) 
%        xlabel(['N sensors ' num2str(sum(z2==i))])  
%         axis([c x_lim c x_lim]) 
%         
%        subplot(2,Nclas,Nclas+i)
       [a,b]=hist(Dist_Test(2:end,i),linspace(0.4,1.2,round(sqrt(Nrandom)/2.2)));
       stairs(b,a/sum(a)), hold on
       plot(Dist_Test(1,i),0,'xr','markersize',20),axis square
       xmean=Mean_Dist(i)-2.576*Std_Dist(i)/sqrt(Nrandom);
       ymean=Mean_Dist(i)+2.576*Std_Dist(i)/sqrt(Nrandom);
       plot([xmean xmean],[0 max(a/sum(a)) ],'b')
       plot([ymean ymean],[0 max(a/sum(a)) ],'b') 
       xlim([0.4 1.2]) 
       ylim([0 0.25]) 
       
       %title(['[' num2str(xmean) ', '  num2str(ymean) ']'])
       if i==1
           set(gca,'Ytick', 0:0.05:0.25,'YtickLabel',0:0.05:0.25,'Xtick',0.4 :0.2:1.2,'XtickLabel',0.4 :0.2:1.2)
       else
           set(gca,'Ytick', [],'YtickLabel',[],'Xtick',0.4 :0.2:1.2,'XtickLabel',0.4 :0.2:1.2)
    
       end
%       
%        Minfo(cont,1)=eval(Vnom{jj}(7));
%        Minfo(cont,2)=eval(Vnom{jj}(10:11));
%        Minfo(cont,3)=Nclas;
%        Minfo(cont,4)=i;
%       
%        Minfo(cont,6)=round(Dist_Test(1,i)*1000)/1000 ;
%        Minfo(cont,7)=round( xmean *1000)/1000;
%        Minfo(cont,8)=round(ymean*1000)/1000; 
%        Minfo(cont,9)=round(pval*100)/100;
%        cont=cont+1;
       if pval<0.005
           text(0.85,0.2,['pval 0.00' ])
       else
           text(0.85,0.2,['pval ' num2str( round(pval*100)/100) ])
       end
     
     end
     %subplot(2,Nclas,Nclas+1)
      subplot(1,Nclas,1)
      ylabel(['P(KLD(S_{rand},S_{Cart}))'])  
%     subplot(2,Nclas,1)
     nom=[Vnom{jj}(7) 'x' Vnom{jj}(10:11) 'pixelNclas'   num2str(Nclas) '_2'];
    % ylabel(nom)
     save2pdf(nom)
     close all
  
end
end