function [Ms,xf1,deltas_xf_ang_phase1,xf2,deltas_xf_ang_phase2,xfm,deltas_xf_ang_phasem,err1,err2,errm] = sort_basis(M,fs)

% SORT_BASIS sorts the column vectors of matrix M according to their
% 2D spatio-frequency meaning
%
% [Ms,xf1,deltas_xf_ang_phase1,xf2,deltas_xf_ang_phase2,xfm,deltas_xf_ang_phasem,err1,err2,errm] = sort_basis(M,fs);

d = length(M(1,:));
D = sqrt(length(M(:,1)));

xf1 = zeros(d,4);
deltas_xf_ang_phase1  = zeros(d,4);
xf2 = zeros(d,4);
deltas_xf_ang_phase2 = zeros(d,4);
xfm = zeros(d,4);
deltas_xf_ang_phasem = zeros(d,4);

[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(D,D,1,fs,fs,1);

err1 = [];
errm = [];
err2 = [];

for i = 1:d
    i
    B = reshape(M(:,i),D,D);
    TFB = fftshift(fft2(B));
    
    % Fit gaussian first and couple of gaussians next
    [m_opt,a_opt,b_opt,alfa_opt,e] = fit_gaussian(B,x,y,0);
    
    xf1(i,1:2) = [m_opt];
    deltas_xf_ang_phase1(i,1:3) = [a_opt b_opt alfa_opt];
    
    [f_opt,dfa_opt,dfb_opt,alfaf_opt,ef] = fit_2_gaussians(TFB,fx,fy,0);

    xf1(i,3:4) = [f_opt];
    
    % Fit Gabor at once
    [m_optm,a_optm,b_optm,alfa_optm,f_optm,phase_optm,em] = fit_gabor(B,x,y,m_opt,f_opt,a_opt,b_opt,alfa_opt,0);
    xfm(i,:) = [m_optm f_optm];
    deltas_xf_ang_phasem(i,:) = [a_optm b_optm alfa_optm phase_optm];

    % Fit only frequency and phase of Gabor (assuming parameters of the spatial Gaussian)
    [m_opt2,a_opt2,b_opt2,alfa_opt2,f_opt2,phase_opt2,e2] = fit_gabor_f_phase_x(B,x,y,m_opt,f_opt,a_opt,b_opt,alfa_opt,0);

    xf2(i,:) = [m_opt2 f_opt2];
        
    deltas_xf_ang_phase1(i,4) = phase_opt2;
    deltas_xf_ang_phase2(i,:) = deltas_xf_ang_phase1(i,:);

    err1 = [err1 (e+ef)/2];
    errm = [errm em];
    err2 = [err2 e2];
    
end 

Ms = M;