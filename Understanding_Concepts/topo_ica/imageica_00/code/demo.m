function demo( exnum )
% demo - example use of the code in this directory
%

%-----------------------------------------------------------
% Gather the patches from the images...
%-----------------------------------------------------------
global X; 
if exnum>=1 & exnum<=3
  % These are large experiments (16-by-16 windows, 160 dimensions)
  [X, whiteningMatrix, dewhiteningMatrix] = data( 50000, 16, 160 );

elseif exnum>=5 & exnum<=7
  % These are smaller experiments (8-by-8 windows, 40 dimensions)
  [X, whiteningMatrix, dewhiteningMatrix] = data( 10000, 8, 40 );

end
  


switch exnum,
  
 case 1,

  %-----------------------------------------------------------
  % LARGE - Standard ICA (simple cell model)
  %-----------------------------------------------------------
    
  p.seed = 1;
  p.write = 5;
  p.model = 'ica';
  p.algorithm = 'fixed-point';
  p.components = 160;
  estimate( whiteningMatrix, dewhiteningMatrix, '../results/ica.mat', p );
  
 case 2,
  
  %-----------------------------------------------------------
  % LARGE - Independent Subspace Analysis (complex cell model)
  %-----------------------------------------------------------
  
  p.seed = 1;
  p.write = 5;
  p.model = 'isa';
  p.algorithm = 'gradient';
  p.groupsize = 4;
  p.groups = 40;
  p.stepsize = 0.1;
  p.epsi = 0.005;
  estimate( whiteningMatrix, dewhiteningMatrix, '../results/isa.mat', p );
  
 case 3,
  
  %-----------------------------------------------------------
  % LARGE - Topographic ICA (model for complex cells and topography)
  %-----------------------------------------------------------
  
  p.seed = 1;
  p.write = 5;
  p.model = 'tica';
  p.algorithm = 'gradient';
  p.xdim = 16;
  p.ydim = 10;
  p.maptype = 'torus';
  p.neighborhood = 'ones3by3';
  p.stepsize = 0.1;
  p.epsi = 0.005;
  estimate( whiteningMatrix, dewhiteningMatrix, '../results/tica.mat', p );
  
 case 4,
  
  %-----------------------------------------------------------
  % LARGE - Displaying the estimated bases
  %-----------------------------------------------------------
  
  load ../results/ica.mat; visual( A, 2, 16 );
  load ../results/isa.mat; visual( A, 2, 16 );
  load ../results/tica.mat; visual( A, 2, 16 );
  
 case 5,

  %-----------------------------------------------------------
  % SMALL - Standard ICA (simple cell model)
  %-----------------------------------------------------------
    
  p.seed = 1;
  p.write = 5;
  p.model = 'ica';
  p.algorithm = 'fixed-point';
  p.components = 40;
  estimate( whiteningMatrix, dewhiteningMatrix, '../results/ica2.mat', p );
  
 case 6,
  
  %-----------------------------------------------------------
  % SMALL - Independent Subspace Analysis (complex cell model)
  %-----------------------------------------------------------
  
  p.seed = 1;
  p.write = 5;
  p.model = 'isa';
  p.algorithm = 'gradient';
  p.groupsize = 2;
  p.groups = 20;
  p.stepsize = 0.1;
  p.epsi = 0.005;
  estimate( whiteningMatrix, dewhiteningMatrix, '../results/isa2.mat', p );
  
 case 7,
  
  %-----------------------------------------------------------
  % SMALL - Topographic ICA (model for complex cells and topography)
  %-----------------------------------------------------------
  
  p.seed = 1;
  p.write = 5;
  p.model = 'tica';
  p.algorithm = 'gradient';
  p.xdim = 8;
  p.ydim = 5;
  p.maptype = 'torus';
  p.neighborhood = 'ones3by3';
  p.stepsize = 0.1;
  p.epsi = 0.005;
  estimate( whiteningMatrix, dewhiteningMatrix, '../results/tica2.mat', p );
  
 case 8,
  
  %-----------------------------------------------------------
  % SMALL - Displaying the estimated bases
  %-----------------------------------------------------------
  
  load ../results/ica2.mat; visual( A, 3, 8 );
  load ../results/isa2.mat; visual( A, 3, 8 );
  load ../results/tica2.mat; visual( A, 3, 8 );

end  
  
