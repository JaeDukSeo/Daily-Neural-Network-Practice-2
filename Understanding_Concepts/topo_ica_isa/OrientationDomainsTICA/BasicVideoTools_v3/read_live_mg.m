function Y_ref = read_live_mg(file_name)

% Y_3d = read_vqeg_mg(path,ind_mov)
%
% file_name = string with the movie name (including the path).
%
% Example for file_name:
% /media/disk/vista/BBDD/VIDEO/VQEG/src2_ref__625.yuv
% 
% Y_3d = 3d array with the sequence of frames of luminance values 
%

width = 768;
height = 432;
data = yuv2mov(file_name, width, height);

Y_ref = zeros(height/2,width/2,length(data));

for ii=1:length(data)
    frame=data(ii).cdata;
    frame = rgb2ycbcr(frame);
    Y_ref(:,:,ii) = double(squeeze(frame(1:2:end,1:2:end,1))); % It skips UV components
end
