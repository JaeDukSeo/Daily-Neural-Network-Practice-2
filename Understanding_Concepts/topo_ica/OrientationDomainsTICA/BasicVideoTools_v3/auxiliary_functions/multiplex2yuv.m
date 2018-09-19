function  YUVtemp = multiplex2yuv(fileName,bound)

% close all
% clear all
% clc
% %fileName = 'src1_ref__625.yuv';bound = 210;
% fileName = 'src15_ref__525.yuv';bound = 250;

fileId = fopen(fileName,'r');

BitStream = fread(fileId);
fclose(fileId);

width = 720;

if bound == 210
    height = 576;
else
    height = 486; 
end

nbr_frames= length(BitStream)/(2*width*height);
YUV = zeros (height,width,3);
YUVtemp = zeros (height,width,3*(nbr_frames - 20));


for idx_fr = 1:nbr_frames
    
    Y = 127*ones(height,width);
    U = Y;
    V = Y;
    
    for idx_line = 1: height
        
        pointer = ( idx_fr - 1)*(2*width*height) + ( idx_line - 1)*width*2;
        
        Yline = BitStream( pointer + 2 : 2 : pointer + 2*width);
        Y(idx_line,:) = Yline;
        
        Uline = BitStream( pointer + 1 : 4 : pointer + 2*width);
        U(idx_line,1:2:end) = Uline;
        
        Vline = BitStream( pointer + 3 : 4 : pointer + 2*width);
        V(idx_line,1:2:end) = Vline;
        
    end
    
    YUV(:,:,1) = Y;
    YUV(:,:,2) = U;
    YUV(:,:,3) = V;

        if (idx_fr >= 11) && (idx_fr <= bound)
            
            [idx_fr (3*(idx_fr-10)-2) 3*(idx_fr-10)];
            
            YUVtemp(:,:,(3*(idx_fr-10)-2) : 3*(idx_fr-10)) = YUV;
            %RGB = ycbcr2rgb(uint8(YUV));
            %Mrgb(idx_fr - 10) = im2frame(RGB);
            
        end

    
    
    %
    %     figure(101)
    %     imagesc(SA)
    
    
    %time=reshape([0.04:0.04:8;0.04:0.04:8;0.04:0.04:8],1,600);
    
    
    
end



% figure
% movie(Mrgb,1,25)