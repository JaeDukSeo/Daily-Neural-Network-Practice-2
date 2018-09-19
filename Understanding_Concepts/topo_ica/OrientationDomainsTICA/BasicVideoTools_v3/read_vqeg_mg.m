function VQEG_mov = read_vqeg_mg(file_name)

%
% Y_3d = read_vqeg_mg(path,ind_mov)
%
% file_name = string with the movie name (including the path).
%
% Example for file_name:
% /media/disk/vista/BBDD/VIDEO/VQEG/src2_ref__625.yuv
% 
% Y_3d = 3d array with the sequence of luminance values 
%        


% fileName = 'src1_ref__625.yuv';bound = 210;
% fileName = 'src15_ref__525.yuv';bound = 250;

if strcmp(file_name(end-6:end-4),'625')
    fileName_ref = [ file_name];
    YUV_ref = multiplex2yuv(fileName_ref,210);
else
    fileName_ref = [ file_name];
    YUV_ref = multiplex2yuv(fileName_ref,250);
end

VQEG_mov = YUV_ref(30:2:end-30,30:2:end-30,1:3:end);
