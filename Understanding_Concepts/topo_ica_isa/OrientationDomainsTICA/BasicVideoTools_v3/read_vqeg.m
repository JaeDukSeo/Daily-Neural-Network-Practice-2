function [YUV_ref,YUV_dist]=read_vqeg(path,ind_peli,want_orig,ind_dist)

% [YUV_orig,YUV_dist]=read_vqeg(path,ind_mov,want_orig,ind_dist)
%
% path      = string with path to the folder with the database
% ind_movie = integer in the range [1:10 13:22]
% ind_dist  = integer in the range 1:16 (zero implies no distorted movie is loaded)
% want_orig = 0 (no) or 1 (yes)
%
% Example for path:
% /media/disk/vista/Ajuste Medida Distorsion/BBDD/VIDEO/VQEG/SDTV/Reference
% 
% CAUTION!: It assumes certain folder organization of the database.
%

tag_video = ind_peli;
idx_dist = ind_dist;

for j = 1:length(tag_video)
    
    idx_video = tag_video(j);
    
    
    % %fileName = 'src1_ref__625.yuv';bound = 210;
    % fileName = 'src15_ref__525.yuv';bound = 250;
    
    
    if (idx_video>=1) && (idx_video<=10)
        sufix = 625;
        
        if want_orig >0
            fileName_ref = [ path,'SDTV/Reference/src' num2str(idx_video) '_ref__' num2str(sufix) '.yuv'];
            YUV_ref = multiplex2yuv(fileName_ref,210);
        end
        
        if ind_dist>0
            
            for k = idx_dist
                fileName_dist = [ path,'SDTV/Distorted/ALL_625/src' num2str(idx_video) '_hrc' num2str(k) '_' num2str(sufix) '.yuv'];
                YUV_dist = multiplex2yuv(fileName_dist,210);
            end
        else
            YUV_dist = 0;
        end
    else
        sufix = 525;
        
        if want_orig >0
            fileName_ref = [ path,'SDTV/Reference/src' num2str(idx_video) '_ref__' num2str(sufix) '.yuv'];
            YUV_ref = multiplex2yuv(fileName_ref,250);
        end
        if ind_dist>0
            for k = idx_dist
                fileName_dist = [ path,'SDTV/Distorted/ALL_525/src' num2str(idx_video) '_hrc' num2str(k) '_' num2str(sufix) '.yuv'];
                YUV_dist = multiplex2yuv(fileName_dist,250);
            end
        else
            YUV_dist = 0;
        end
    end
    
end
