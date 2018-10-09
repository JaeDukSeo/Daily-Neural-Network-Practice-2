%
% GENERATE_DATA_TICA_NO_ALIASING.M
%
% In this script, 
%     (1) We set certain undersampling factor
%     (2) We apply it to the images of the imageica toolbox database 
%     (3) We take vectorized blocks of certain sizes from these images to generate datasets to train ICA
%     (4) We save the sets of vectors in files to be used later.
%         One file per undersampling factor, block size, and image...
%         save([path_result,'data_'  int2str(submestreo) '_',num2str(lado),'_im_',num2str(i) ,'_A'],'xx')
% Warning! paths are hardcoded
% 
%

lados = [16 20 32 50 64 100];
%path_result = '/media/disk/vista/Papers/MODVIS_2016/MODVIS_linear/VisRes_16/code/vector_images/';

path_result = '/media/disk/vista/Papers/PLOS_2016_tica/code/vector_images/';


% for j=4%1:length(lados)
%     lado = lados(j);
%     for i=1:13
%         clear x;
%         imm = double(imread([num2str(i),'.tiff']));
%         x = im2col(imm,[lado lado],'sliding');
%         l = length(x(1,:));
%         ind = randperm(l);
%         xx = x(:,ind(1:10000));
%         save([path_result,'data_',num2str(lado),'_im_',num2str(i)],'xx')
%         [lado i]
%     end
% end
% 
% for j=1:4%length(lados)
%     lado = lados(j);
%     for i=1:13
%         clear x;
%         imm = double(imread([num2str(i),'.tiff']));
%         imm = imm(1:2:end,1:2:end);
%         s = size(imm);
%         if s(1)>s(2)
%            imm = [imm imm(:,end:-1:1)]; 
%            imm = [imm imm;imm imm];
%         else
%            imm = [imm;imm(end:-1:1,:)]; 
%            imm = [imm imm;imm imm];
%         end        
%         x = im2col(i/mm,[lado lado],'sliding');
%         l = length(x(1,:));
%         ind = randperm(l);
% %         if lado <100
% %            xx = x(:,ind(1:10000));
% %         else 
% %            xx = x(:,ind(1:4500));
% %         end
%         xx = x(:,ind(1:10000));
%         save([path_result,'data_2x_',num2str(lado),'_im_',num2str(i)],'xx')
%         [lado i]
%     end
% end

% submuestreando cortando posiciones: sin antialiasing
% for j=1:length(lados)
%     lado = lados(j)
%     for i=5:13
%         clear x;
%         imm = double(imread([num2str(i),'.tiff']));        
%         imm = imm(1:4:end,1:4:end);
%         s = size(imm);
%         if s(1)>s(2)
%            imm = [imm imm(:,end:-1:1)]; 
%            imm = [imm imm;imm imm];
%         else
%            imm = [imm;imm(end:-1:1,:)]; 
%            imm = [imm imm;imm imm];
%         end
%         x = im2col(imm,[lado lado],'sliding');
%         l = length(x(1,:));
%         ind = randperm(l);
% %         if lado <100
% %            xx = x(:,ind(1:10000));
% %         else 
% %            xx = x(:,ind(1:10000));
% %         end
%         xx = x(:,ind(1:10000));
%         save([path_result,'data_4x_',num2str(lado),'_im_',num2str(i) ],'xx')
%         [lado s i]
%     end
% end

% submuestreando imresize: con antialiasing
submestreo=2
for j=1:length(lados)
    lado = lados(j)
    for i=1:13
        clear x;
        imm = double(imread([num2str(i),'.tiff']));
        imm = imresize(imm ,[size(imm)/submestreo]);
        
        s = size(imm);
        if s(1)>s(2)
           imm = [imm imm(:,end:-1:1)]; 
           imm = [imm imm;imm imm];
        else
           imm = [imm;imm(end:-1:1,:)]; 
           imm = [imm imm;imm imm];
        end
        x = im2col(imm,[lado lado],'sliding');
        l = length(x(1,:));
        ind = randperm(l);
%         if lado <100
%            xx = x(:,ind(1:10000));
%         else 
%            xx = x(:,ind(1:10000));
%         end
        xx = x(:,ind(1:10000));
        save([path_result,'data_'  int2str(submestreo) '_',num2str(lado),'_im_',num2str(i) ,'_A'],'xx')
        [lado s i] 
    end
end