import glob
import moviepy.editor as mpy

gif_name = 'gif'; fps = 5

file_list = glob.glob('gradientp/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('.')[0].split('\\')[1] )) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps).resize(0.3)
clip.write_gif('gradientp/{}.gif'.format(gif_name), fps=fps)

file_list = glob.glob('gradientw/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('.')[0].split('\\')[1] )) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps).resize(0.3)
clip.write_gif('gradientw/{}.gif'.format(gif_name), fps=fps)

file_list = glob.glob('layer/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('.')[0].split('\\')[1] )) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps).resize(0.3)
clip.write_gif('layer/{}.gif'.format(gif_name), fps=fps)

file_list = glob.glob('layera/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('.')[0].split('\\')[1] )) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps).resize(0.3)
clip.write_gif('layera/{}.gif'.format(gif_name), fps=fps)

file_list = glob.glob('moment/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('.')[0].split('\\')[1] )) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps).resize(0.3)
clip.write_gif('moment/{}.gif'.format(gif_name), fps=fps)

file_list = glob.glob('weights/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('.')[0].split('\\')[1] )) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps).resize(0.3)
clip.write_gif('weights/{}.gif'.format(gif_name), fps=fps)