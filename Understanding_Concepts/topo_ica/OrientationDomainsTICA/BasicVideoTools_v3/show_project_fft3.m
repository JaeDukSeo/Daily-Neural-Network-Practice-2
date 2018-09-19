function show_project_fft3(tf,fse,fst,fig)

% SHOW_PROJECT_FFT3 shows the three projections of a 3D fft
% 
% It assumes real values and square frames. The transform has to be in 2d
% array format
%
% SYNTAX: show_project_fft3(F,fsx,fst,figure)
%

tf=abs(tf);

tfxy=proytf3(tf,1);
tfxt=proytf3(tf,2);
tfyt=proytf3(tf,3);

m=size(tf);
nf=m(2)/m(1);
fe=linspace(-fse/2,fse/2,m(1));
ft=linspace(-fst/2,fst/2,nf);

figure(fig);image(fe,fe,64*tfxy/maxi(tfxy)),ax,xlabel('fx (c/deg)'),ylabel('fy (c/deg)')
figure(fig+1);image(ft,fe,64*tfxt/maxi(tfxt)),ax,xlabel('ft (Hz)'),ylabel('fx (c/deg)')
figure(fig+2);image(ft,fe,64*tfyt/maxi(tfyt)),ax,xlabel('ft (Hz)'),ylabel('fy (c/deg)')

[maxi(tfxy) maxi(tfxt) maxi(tfyt)]