
clear all
close all

toolbox_folder = '/media/disk/vista/Software/video_software/';

addpath(genpath(toolbox_folder))


% VQEG

mov_VQEG = read_vqeg_mg([toolbox_folder 'illustrative_video_data/VQEG/src3_ref__625.yuv']);


% LIVE
mov_LIVE = read_live_mg([toolbox_folder 'illustrative_video_data/LIVE/bs1_25fps.yuv']);
            
            
% 

implay(mov_VQEG./255)

implay(mov_LIVE./255)
