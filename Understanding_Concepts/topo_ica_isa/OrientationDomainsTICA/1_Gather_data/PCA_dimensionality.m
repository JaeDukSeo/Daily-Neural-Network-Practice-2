%
% PCA_DIMENSIONALITY.M
%
% In this script I analyze the (99%) PCA dimensionality of the 
% image datasets considered in the work.
% It assumes that generate_data_tica_no_aliasing.m has been executed to
% generate the vectors. It assumes the undersampling explored at the work
% (warning! names of the files to be read are hardcoded)
% It saves the computed PCA domensionalities for different percentages of energy and different undersampling factors.
%
% save([path_result,'PCA_dimensionality'],'dim*')


lados = [16 20 32 50 100];
path_result = 'C:\disco_portable\mundo_irreal\latex\Orientation_Domains_TICA\code';

dim99 = zeros(1,5);
dim95 = zeros(1,5);
dim90 = zeros(1,5);
dim85 = zeros(1,5);
for j=1:length(lados)
    lado = lados(j);
    C = zeros(lado*lado,lado*lado);
    for i=1:13
        load([path_result,'data_',num2str(lado),'_im_',num2str(i)],'xx');
        C = C + xx*xx';
        [lado i]
    end
    clear xx
    tic
    [B,L]=eig(C);
    toc
    L = abs(diag(L));
    L = sort(L,'descend');
    cL = cumsum(L);
    cL = 100*cL/cL(end);
    dim99(j) = interp1(cL,1:lado*lado,99)
    dim95(j) = interp1(cL,1:lado*lado,95)
    dim90(j) = interp1(cL,1:lado*lado,90)
    dim85(j) = interp1(cL,1:lado*lado,85)
end

lados = [32 50 100];
dim99_2x = zeros(1,3);
dim95_2x = zeros(1,3);
dim90_2x = zeros(1,3);
dim85_2x = zeros(1,3);
for j=1:length(lados)
    lado = lados(j);
    C = zeros(lado*lado,lado*lado);
    for i=1:6
        load([path_result,'data_2x_',num2str(lado),'_im_',num2str(i)],'xx');
        C = C + xx*xx';
        [lado i]
    end
    clear xx
    tic
    [B,L]=eig(C);
    toc
    L = abs(diag(L));
    L = sort(L,'descend');
    cL = cumsum(L);
    cL = 100*cL/cL(end);
    dim99_2x(j) = interp1(cL,1:lado*lado,99)
    dim95_2x(j) = interp1(cL,1:lado*lado,95)
    dim90_2x(j) = interp1(cL,1:lado*lado,90)
    dim85_2x(j) = interp1(cL,1:lado*lado,85)
end

lados = [32 50 100];
dim99_4x = zeros(1,3);
dim95_4x = zeros(1,3);
dim90_4x = zeros(1,3);
dim85_4x = zeros(1,3);
for j=1:length(lados)
    lado = lados(j);
    C = zeros(lado*lado,lado*lado);
    for i=1:6
        load([path_result,'data_4x_',num2str(lado),'_im_',num2str(i)],'xx');
        C = C + xx*xx';
        [lado i]
    end
    clear xx
    tic
    [B,L]=eig(C);
    toc
    L = abs(diag(L));
    L = sort(L,'descend');
    cL = cumsum(L);
    cL = 100*cL/cL(end);
    dim99_4x(j) = interp1(cL,1:lado*lado,99)
    dim95_4x(j) = interp1(cL,1:lado*lado,95)
    dim90_4x(j) = interp1(cL,1:lado*lado,90)
    dim85_4x(j) = interp1(cL,1:lado*lado,85)
end

save([path_result,'PCA_dimensionality'],'dim*')

%%%%%%%%%%%%%%% 99

angulos = [0.64 0.8 1.28 2 1.28*2 4 1.28*4 2*4];
dim_99 = [dim99(1) dim99(2) dim99(3) dim99(4) dim99_2x(1) dim99_2x(2) dim99_4x(1) dim99_4x(2)];
dim_99_100 = dim99(5);
dim_95 = [dim95(1) dim95(2) dim95(3) dim95(4) dim95_2x(1) dim95_2x(2) dim95_4x(1) dim95_4x(2)];
dim_95_100 = dim95(5);

figure,semilogy(angulos,dim_99,'bo-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
hold on,semilogy(4,dim_99_100,'b*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
hold on,semilogy(4,dim_99_100,'bo'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
axis([0 10 30 2500])
fact = 0.3;
text(angulos(1)+fact,dim_99(1),'16x16 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(2)+fact,dim_99(2),'20x20 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(3)+fact,dim_99(3),'32x32 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(4)+fact,dim_99(4),'50x50 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(5)+fact,dim_99(5),'32x32 (f_s = 12.5 cpd)','color',[0 0 1])
text(angulos(6)+fact,dim_99(6),'50x50 (f_s = 12.5 cpd)','color',[0 0 1])
text(angulos(7)+fact,dim_99(7),'32x32 (f_s = 6.25 cpd)','color',[0 0 1])
text(angulos(8)+fact,dim_99(8),'50x50 (f_s = 6.25 cpd)','color',[0 0 1])
text(angulos(6)+fact,dim_99_100,'100x100 (f_s = 25 cpd)','color',[1 0 0])
set(gcf,'color',[1 1 1])

[dim_99 dim_99_100]/dim_99(3)

%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%

angulos = [0.64 0.8 1.28 2 1.28*2 4 1.28*4 2*4];
dim_99 = [dim99(1) dim99(2) dim99(3) dim99(4) dim99_2x(1) dim99_2x(2) dim99_4x(1) dim99_4x(2)];
dim_99_100 = dim99(5);
dim_95 = [dim95(1) dim95(2) dim95(3) dim95(4) dim95_2x(1) dim95_2x(2) dim95_4x(1) dim95_4x(2)];
dim_95_100 = dim95(5);

figure,plot(angulos,dim_99,'bo-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
hold on,plot(4,dim_99_100,'b*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
hold on,plot(4,dim_99_100,'bo'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
axis([0 10 30 2500])
fact = 0.3;
text(angulos(1)+fact,dim_99(1),'16x16 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(2)+fact,dim_99(2),'20x20 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(3)+fact,dim_99(3),'32x32 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(4)+fact,dim_99(4),'50x50 (f_s = 25 cpd)','color',[0 0 1])
text(angulos(5)+fact,dim_99(5),'32x32 (f_s = 12.5 cpd)','color',[0 0 1])
text(angulos(6)+fact,dim_99(6),'50x50 (f_s = 12.5 cpd)','color',[0 0 1])
text(angulos(7)+fact,dim_99(7),'32x32 (f_s = 6.25 cpd)','color',[0 0 1])
text(angulos(8)+fact,dim_99(8),'50x50 (f_s = 6.25 cpd)','color',[0 0 1])
text(angulos(6)+fact,dim_99_100,'100x100 (f_s = 25 cpd)','color',[1 0 0])
set(gcf,'color',[1 1 1])

%%%%%%%%%%%%%%%%%%

figure,semilogy([0.64 0.8 1.28 2 4],dim99,'bo-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy([0.64 0.8 1.28 2 4],[16 20 32 50 100].^2,'b*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,semilogy(2*[1.28 2 4],dim99_2x,'ko-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(2*[2 4],[50 100].^2,'k*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,semilogy(4*[1.28 2 4],dim99_4x,'mo-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(4*[2 4],[50 100].^2,'m*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
h=legend('f_s = 25.00 cpd','f_s = 12.50 cpd','f_s =  6.25 cpd')                                             
set(h,'box','off','location','southeast')

figure,semilogy(angulos,dim,'ko-')

%%%%%%%%%%%%%%%%%%%%%%%%%

figure,plot([0.64 0.8 1.28 2 4],dim99,'bo-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy([0.64 0.8 1.28 2 4],[16 20 32 50 100].^2,'b*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,plot(2*[1.28 2 4],dim99_2x,'ko-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(2*[2 4],[50 100].^2,'k*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,plot(4*[1.28 2 4],dim99_4x,'mo-'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(4*[2 4],[50 100].^2,'m*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
h=legend('f_s = 25.00 cpd','f_s = 12.50 cpd','f_s =  6.25 cpd')                                             
set(h,'box','off','location','southeast')

%%%%%%%%%%%%%%% 95

figure,semilogy([0.64 0.8 1.28 2 4],dim95,'bo-'),xlabel('Visual Angle'),ylabel('95% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy([0.64 0.8 1.28 2 4],[16 20 32 50 100].^2,'b*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,semilogy(2*[1.28 2 4],dim95_2x,'ko-'),xlabel('Visual Angle'),ylabel('95% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(2*[2 4],[50 100].^2,'k*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,semilogy(4*[1.28 2 4],dim95_4x,'mo-'),xlabel('Visual Angle'),ylabel('95% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(4*[2 4],[50 100].^2,'m*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
h=legend('f_s = 25.00 cpd','f_s = 12.50 cpd','f_s =  6.25 cpd')                                             
set(h,'box','off','location','southeast')

%%%%%%%%%%%%%%%%%%

figure,semilogy([0.64 0.8 1.28 2 4],dim90,'bo-'),xlabel('Visual Angle'),ylabel('90% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy([0.64 0.8 1.28 2 4],[16 20 32 50 100].^2,'b*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,semilogy(2*[1.28 2 4],dim90_2x,'ko-'),xlabel('Visual Angle'),ylabel('90% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(2*[2 4],[50 100].^2,'k*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
                                             
hold on,semilogy(4*[1.28 2 4],dim90_4x,'mo-'),xlabel('Visual Angle'),ylabel('90% PCA dimensionality')
                                             title(['Intrinsic Dimension vs Visual Angle'])
% hold on,semilogy(4*[2 4],[50 100].^2,'m*'),xlabel('Visual Angle'),ylabel('99% PCA dimensionality (f_s = 25 cpd)')
%                                              title(['Intrinsic Dimension vs Visual Angle'])
h=legend('f_s = 25.00 cpd','f_s = 12.50 cpd','f_s =  6.25 cpd')                                             
set(h,'box','off','location','southeast')
