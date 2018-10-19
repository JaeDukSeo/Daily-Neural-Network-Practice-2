function  [S_sup_b,S_surr_b,S_t,S_sup,S_surr,S_test,apert] = combine_test_background(imt,ct,imb,cb,L0,radius,sharp)

% COMBINE_TEST_BACKGROUND combines two images in a center-surround setting 
% with and without superposition, and controlling contrasts and average luminance.
% The relative width of the center and the sharpness of the blending can be controlled too. 
% Images are asummed to be squared and the same size.  
%
%   [S_sup_b,S_surr_b,S_t,S_sup,S_surr,S_test,apert] = combine_test_background(imt,ct,imb,cb,L0,radius,sharp)
%
%       imt = image in the center (test)
%        ct = contrast of test 
%       imb = image in the surround (background) 
%        cb = contrast of background 
%        L0 = average luminance
%    radius = relative radius (1 means size/2) 
%     sharp = central region is defined with a generalized-gaussian. The sharpness is the exponent of this supergaussian.
%             (sharp = 2 means gaussian, and the bigger the sharper)
%
%    S_sup_b   = background (zero mean)
%    S_surr_b  = surround (zero mean)
%    S_t       = test (zero mean)
%    S_sup     = background (with nonzero mean)
%    S_surr    = surround (on top of average luminance)
%    S_test    = test on top of average luminance
%    apert     = aperture
%

s1 = size(imt);
s2 = size(imb);

if (s1(1)==s1(2))&(s2(1)==s2(2))&(s1(1)==s2(1))

    % Relative domain
    %-----------------
    s1 = s1(1);
    nx = s1;
    fsx = s1;
    num_frames = 1;
    fst = 1;
    [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(nx,nx,num_frames,fsx,fsx,fst);

    % Circular aperture
    % -----------------
    
    %e = 14;      % Sharpness
    %width = 0.7; % Radius
    apert = exp( -((sqrt(   (x-0.5).^2 + (y-0.5).^2  ).^sharp)/(radius/2)^sharp  ));


    % Contrast of test 
    st = control_lum_contrast(imt,L0,ct);
    st = (st-L0);
    
    % Contrast of background 
    sb = control_lum_contrast(imb,L0,cb);
    sb = (sb-L0);
    
    % Test
    % -------------
    
    S_t = apert.*st;

    % Isolated test
    
    S_test = S_t + L0;
    
    % Superposition
    % ------------------

    S_sup_b = sb + L0;

    S_sup = S_sup_b + S_t; 
    
    % Surround
    % ------------------

    S_surr_b = (1-apert).*sb  + L0;

    S_surr = S_surr_b + S_t; 

else
    disp(' ')
    disp('  Nothing was done!: center and surround images should be squared and equal in size!')
    disp(' ')
end