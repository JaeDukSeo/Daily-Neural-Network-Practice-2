% INPUT
%    Sortu files
%    PARAMETERS TO DEFINE:    
%    Nclas           Integrer, Number of sets to divide the angles
%    Nrandom         Integrer, Boostrap elements
%    GrauIn          Origin in degrees, to make the sets on classes
%    bordes          
%    plot_on         Boolean, 1 plots the reuslts
%    dist            Mesure of distance used, options:
%                   'KL':            Kullback–Leibler divergence
%                   'bhattacharyya': Bhattacharyya distance
%                   'empty_cells':   Number of empty cells          
 

%-----------------------------------
% OUTPUT
% Plots showing the results
% Fig7 Locations of the TICA sensory units in the retinal space depending 
% on their preferred orientation (sets S ori), and uniformity test results.  
% In order to point out the consistency of the result regardless of the 
% arbitrary partition of the orientation range, here we apply the test
% splitting the data (the sensory units) according to three different 
% partitions of the [0,180] o ] range.
% P_values results 
% -----------------------------------
% VARIABLES TO SET 
clear all
%close all
path_read='C:\Users\marina\Desktop\UV_IPL\Projectes\ICA_Pinwheels_Hyva\Codi\Results_ICA';
Nrandom=10000;
GrauIn=0;
bordes=0.1;
plot_on=1;
dist='KL';
%dist='bhattacharyya';
%dist='empty_cells';         
%-----------------------------------
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
Vfs([6 8])=12;
Vfs([7 9])=6;
%-----------------------------------
% visualize TICA functiones
% A, W, p.xdim dimencion of the TICA functions
% colormap gray, disp_patches(W',p.xdim,2 );
% colormap gray, disp_patches(A,p.xdim,2 );
% imagesc(reshape(A(11,:), [p.xdim p.ydim]))% 
% imagesc(reshape(A(:,10), [sqrt(size(A,1)) sqrt(size(A,1))])) 
% imagesc(reshape(W(:,100), [p.xdim p.ydim]))% 
% imagesc(reshape(W(10,:), [sqrt(size(A,1)) sqrt(size(A,1))]))

ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);

%-----------------------------------
% MAIN LOOP: The number of classes to discretise the angles
for Nclas=[4]% 5 6]
    Ngrau=180/Nclas;

    % for the 7 TICA sets we have
    for jj=2:9
        Vnom{jj};
        %load(Vnom{jj})
        fs=Vfs(jj);
        load([path_read '/' Vnom{jj}  ])
        [nfA,lala]=size(Mss);
        N=sqrt(nfA);
        
        xf_sort=xf2_t;
        z=round(359*(pi+atan2(xf_sort(:,4),xf_sort(:,3)))/(2*pi))+1;
        x=xf_sort(:,1);
        y=xf_sort(:,2);
        c=N/fs*bordes;
        xmin=c;
        xmax= N/fs-c;
        %----------------------------------
        % square around the retinal field
        InSquare=(x>c & x<=xmax  & y<=xmax & y>c);
        x=x(InSquare);
        y=y(InSquare );
        z=z(InSquare );
        Nsen=length(y); % sensors in the scuare
        %-----------------------------------
        % Discretised the continuos set of angles in Nclas sets
        Graus=Ngrau/2;
        VGRaus=GrauIn:Ngrau:180-Graus;
        z(z>180)=z(z>180)-180;
        z2=zeros(size(z));
        
        VGRausS=VGRaus+Graus;
        VGRausI=VGRaus-Graus;
        z2=ones(size(z));
        for i= 1:length(VGRaus)
            if i==1
                z2(0<z & z<=VGRausS(i))=i;
                z2(VGRausS(end)<z & z<=180)=i;
            else
                z2(VGRausI(i)<z & z<=VGRausS(i))=i;
            end
        end
        
        % checks --------------------------------------------------------------
        if  sum(z2==0)
            error
        end
        NclasMean=round(Nsen/Nclas);
        for i=1:Nclas
            V_clas(jj,i)=sum(z2==i);
        end
        %mean(V_clas )
        %---------------------------------------------------------------------
        % TEST
        Dist_Test=zeros(Nrandom,Nclas);
        for i=1:Nclas
            NsenSq1=ceil(sqrt(sum(z2==i)));
            NsenSq2=floor(sqrt(sum(z2==i)));
            Dist_Test(1,i)=Mdist3([x(z2==i),y(z2==i)], c,xmax,[NsenSq1,NsenSq2],dist);
        end
        %---------------------------------------------------------------------
        % Boostrap part
        for k=1: Nrandom
            xr=rand(Nsen,1)*(N/fs-2*c)+c;
            yr=rand(Nsen,1)*(N/fs-2*c)+c;
            for i=1:Nclas
                NsenSq1=ceil(sqrt(sum(z2==i)));
                NsenSq2=floor(sqrt(sum(z2==i)));
                Dist_Test(k+1,i)=Mdist3([xr(z2==i),yr(z2==i)], c,xmax,[NsenSq1,NsenSq2],dist);
                % limites
            end
        end
        %---------------------------------------------------------------------
        % pvalues
        for i=1:Nclas
            MTest_Dist(jj,i)=sum(Dist_Test(1,i)>Dist_Test(2:end,i))/Nrandom;
        end
        %---------------------------------------------------------------------
        % mean int
        Mean_Dist= mean(Dist_Test(2:end,:));
        Std_Dist= std(Dist_Test(2:end,:));
        %--------------------------------------------------------------------
        % PLOT
        if plot_on
            for i=1:Nclas
                figure(jj)
                subplot(2,Nclas,i)
                Vindex=xmin:(xmax-xmin)/(NsenSq1):xmax;
                plot(x(z2==i),y(z2==i),'.b','markersize',20),axis square,axis ij, hold on
                set(gca,'Xtick', Vindex,'XtickLabel',[],'Ytick', Vindex,'YtickLabel',[])
                grid on
                axis([c xmax c xmax])
                %title([ num2str(floor(Dist_Test(1,i)*100)/100) '%'   ])
                
                subplot(2,Nclas,Nclas+i)
                [a,b]=hist(Dist_Test(2:end,i),round(sqrt(Nrandom)/4));
                stairs(b,a/sum(a)), hold on
                plot(Dist_Test(1,i),0,'xr')
                xmean=Mean_Dist(i)-2.576*Std_Dist(i)/sqrt(Nrandom);
                ymean=Mean_Dist(i)+2.576*Std_Dist(i)/sqrt(Nrandom);
                plot([xmean xmean],[0 max(a/sum(a)) ],'b')
                plot([ymean ymean],[0 max(a/sum(a)) ],'b')
                %xlim([0.4 0.9])
                ylim([0  max(a/sum(a))*1.1])
                title(['pval ' num2str(floor(MTest_Dist(jj,i)*100)/100)])
                ylabel(['N clase ' num2str(V_clas(jj,i))])
            end
            %         figure(jj)
            %         for kk=7:9
            %             subplot(3,3,kk)
            %             NsenSq=floor(mean (V_clas(jj,:)));
            %             Vindex=xmin:(xmax-xmin)/(NsenSq):xmax;
            %             xr=rand(NclasMean,1)*(N/fs-2*c)+c;
            %             yr=rand(NclasMean,1)*(N/fs-2*c)+c;
            %             plot(xr,yr,'.g','markersize',20),axis square,axis ij, hold on
            %             set(gca,'Xtick', Vindex,'XtickLabel',[],'Ytick', Vindex,'YtickLabel',[])
            %             grid on
            %             aux=Mdist3([xr,yr], c,xmax,[NsenSq1,NsenSq2],dist);
            %             ylabel(['Np ' num2str(NclasMean) ', N sq' num2str(NsenSq.^2)])
            %             xlabel([ num2str(floor(aux*100)/100) '%'   ])
            %             axis([c xmax c xmax])
            %         end
        end
        nom= [dist '_' Vnom{jj}(7:end-4) '_Nclas' num2str(Nclas) '_Nrand_' num2str(Nrandom) '_ori' num2str(floor(GrauIn))];
        save(nom,'Dist_Test')
        clear Dist_Test
        
    end
    MTest_Dist
end