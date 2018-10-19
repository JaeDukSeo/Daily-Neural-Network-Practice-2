function [g,varargout] = ndgauss(hsize,sigma,varargin)
% NDGAUSS: create ND gaussian kernel in any derivative order.
%
%   g = ndgauss(hsize,sigma);
%   [g,xi,yi,..] = ndgauss(hsize,sigma);
%
% Inputs:
%   - hsize is N-length of kernel size.
%   - sigma is N-length of standard deviations (gaussian widths).
%
% Outputs:
%   - g is the kernel. The size of g is hsize(1) x hsize(2) x ... hsize(n).
%   - [xi,yi,..] is the grid to create the kernel.
%
% Options:
% 1. 'der', <gaussian_derivatives>. Default is 0.
%    The option must be in N-length array. Each defines which derivative
%    order.
%    Example:
%    a. to create d^2G(x,y) / dxdy, then 'der' = [1 1].
%    b. to create d^2G(x,y) / dy^2, then 'der' = [0 2].
%
% 2. 'normalize', 1 | 0. Default is 0.
%    If this option is set to 1, the kernel will be divided by its sum that
%    makes the sum of kernel is 1.
%    Warning: gaussian derivative kernel may not be sum to 1, e.g.,
%    dG(x,y)/dxdy.
%
% Examples:
% 1. To create 33x33 2D gaussian kernel with different sigma:
%    zi = ndgauss([33 33],[2*sqrt(2) 1.2]);
%    imagesc(zi); colormap(gray); axis image;
%
% 2. To create several 1D gaussian kernel in different derivative orders:
%    leg = cell(1,5);
%    for i=0:4
%        yi(:,i+1) = ndgauss(33,2*sqrt(2),'der',i);
%        leg{i+1} = sprintf('g%d(x)',i);
%    end
%    plot(yi); 
%    legend(leg);
%
% 3. To create 15x15 Laplacian of Gaussian of width 2:
%    Lg = ndgauss([15 15],[2 2],'der',[2 0]) + ...
%         ndgauss([15 15],[2 2],'der',[0 2]);
%    figure; colormap(gray);
%    subplot(1,2,1); imagesc(Lg); axis image;
%    subplot(1,2,2); surf(Lg); axis vis3d;
%
% Authors:
%   Avan Suinesiaputra - avan dot sp at gmail dot com.
%   Fadillah Z Tala - fadil dot tala at gmail dot com.
% Modified by Jesus Malo (fixed the absolute value issue in the normalization -line 122-)
%

% rev:
% 27/06/2010 - first creation.

% default options
opt.der = [];
opt.normalize = 0;

% get options
for i=1:2:length(varargin)
    if( isfield(opt,varargin{i}) ), opt.(varargin{i}) = varargin{i+1};
    else error('Unknown found.'); 
    end
end

% check der & sigma
if( isempty(opt.der) ), opt.der = zeros(size(hsize)); end

% check sizes
dim = length(hsize);
if( dim ~= length(sigma) ) 
    error('Dimension mismatches.'); 
end

% check derivative order
if( any(opt.der)<0 || any(opt.der-fix(opt.der))~=0 )
    error('Derivative is invalid.'); 
end

% check values of hsize & sigma
if( any(hsize<=0) ), error('One of the kernel size is negative.'); end
if( any(sigma<=0) ), error('One of the sigma is negative.'); end

% half kernel size
sz = (hsize - 1)/2;

% create N-dimensional grid
if( 1==dim )
    X = (-sz:sz)';
    varargout = X;
else
    sa = ''; sr = ''; T = {};
    for i=1:dim
        sa = sprintf('%s,%f:%f',sa,-sz(i),sz(i));
        sr = sprintf('%s,T{%d}',sr,i);
    end
    eval(sprintf('[%s] = ndgrid(%s);',sr(2:end),sa(2:end)));
    X = zeros(numel(T{1}),dim);
    for i=1:dim
        X(:,i) = reshape(T{i},[],1);
    end
    varargout = T;
    clear sa sr T;
end

% normalized 1D gaussian function
gfun = @(x,s) exp(-(x.*x)/(2*s.*s)) ./ (sqrt(2*pi)*s);

% create kernel
for i=1:dim
    c = sigma(i) * sqrt(2);
    gx = gfun(X(:,i),sigma(i));
    gx(gx<eps*max(X(:,i))) = 0;
    Hn = hermite(opt.der(i),X(:,i) ./ c);
    X(:,i) = Hn .* gx .* (-1/c).^opt.der(i);
end
g = prod(X,2);

% normalize kernel, but derivative kernel may not sum to 1.
if( opt.normalize )
    sumg = sum(abs(g));
    if( 0~=sumg )
        g = g ./ sumg;
    end
end

% reshape kernel
if( dim>1 )
    g = reshape(g,hsize);
end