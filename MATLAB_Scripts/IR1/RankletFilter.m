% Authors: Fabrizio Smeraldi: http://www.eecs.qmul.ac.uk/~fabri
%          George Azzopardi : http://www.cs.rug.nl/~george
% Date: April 2009  -  URL  : http://www.ranklets.net
% Free for non-commercial use, all other rights reserved
%
% Efficiently Compute Intensity Ranklets using Distribution Count 
% sorting
%
% See: F. Smeraldi, Fast algorithms for the computation of Ranklets, 
% Proceedings of ICIP, pp 3969-3972, 2009
%
% Inputs:
%    1. imgFilename - The filename of the image to be processed
%    2. xhalfw      - The half width of the kernel; e.g. 5
%    3. yhalfw      - The half height of the kernel; e.g. 5
% Outputs:
%    1. vertical    - The response for the vertically tuned filter
%    2. horizontal  - The response for the horizontally tuned filter
%    3. diagonal    - The response for the diagonally tuned filter

function [vertical, horizontal, diagonal] = RankletFilter(imgFilename, xhalfw, yhalfw)

% Read image file and determine its size
img_info = imfinfo(imgFilename);
img = imread(imgFilename);
if strcmp(img_info.ColorType, 'truecolor') == 1
    img = rgb2gray(img);
end
[xsize,ysize] = size(img);

% Initialize the responses
horizontal = zeros(size(img));
vertical   = zeros(size(img));
diagonal   = zeros(size(img));

% Filter the image with 3 filter orientations
for x = xhalfw+1:xsize - xhalfw
  for y = yhalfw+1:ysize - yhalfw
    [vertical(y,x),horizontal(y,x),diagonal(y,x)] = ranklet(x,y,xhalfw,yhalfw,img);
  end
end

% Computes the ranklet value for the given window for 3 orientations
% Inputs:
%    1. x0, y0 - The coordinate of the middle pixel in the current window
%    2. xhalfw - The half width of the kernel
%    3. yhalfw - The half height of the kernel
%    4. img    - The image
% Outputs:
%    1. vertrk  - The ranklet value for the vertically tuned filter
%    2. horizrk - The ranklet value for the horizontally tuned filter
%    3. diagrk  - The ranklet value for the diagonally tuned filter
function [vertrk, horizrk, diagrk] = ranklet(x0, y0, xhalfw, yhalfw, img)
  
% entire image window
window = double(img(x0 - xhalfw:x0 + xhalfw - 1, y0 - yhalfw:y0 + yhalfw - 1));

% compute histogram between min and max intensity values only 
imin     = min(min(window));
imax     = max(max(window));
if imin == imax
    if imin > 1
        imin = imin - 1;
    elseif imax < 255
        imax = imax + 1;
    end
end
dynrange = double(imin:imax);
histo    = hist(window(:), dynrange);

% discard zeros in histogram and compact it
support     = histo > 0;
histo       = histo(support);
nonzerobins = sum(support);

% Midranks calculation matrix - this has got elements
%   [ 1/2  0    0    0 ...]
%   [ 1/2  1/2  0    0 ...]
%   [ 0    1/2  1/2  0 ...] 
% and so on
midm = 1/2 * (diag(ones(nonzerobins,1)) + diag(ones(nonzerobins-1,1),-1));
cumhisto = cumsum(histo)'; % Vertical vector
midranks = midm(1:nonzerobins,1:nonzerobins) * cumhisto + 0.5;

% Keep only nonzero bins for next histograms
range=dynrange(support);

% Compute T (Treatment) sets:

% Vertical
window = double(img(x0-xhalfw:x0-1, y0-yhalfw:y0+yhalfw-1));
vt     = hist(window(:), range);

% Horizontal
window = double(img(x0-xhalfw:x0+xhalfw-1, y0-yhalfw:y0-1));
ht     = hist(window(:), range);

% Diagonal
window  = double(img(x0-xhalfw:x0-1, y0-yhalfw:y0-1));
window2 = double(img(x0:x0+xhalfw-1, y0:y0+yhalfw-1));
dt = hist(window(:), range) + hist(window2(:), range);


% Scalar product of histogram of test set
% with midranks computes Wilcoxon statistics
vertrk  = vt * midranks;
horizrk = ht * midranks;
diagrk  = dt * midranks;

% Normalize
mean    = xhalfw*yhalfw*(4*xhalfw*yhalfw+1);
scaling = 2* (xhalfw*yhalfw) * (xhalfw * yhalfw);

v_sum = sum(sum(vertrk));
v_num = size(vertrk, 1) * size(vertrk, 1);
vertrk  = (v_sum/v_num - mean) / scaling;

h_sum = sum(sum(horizrk));
h_num = size(horizrk, 1) * size(horizrk, 1);
horizrk = (h_sum/h_num - mean) / scaling;

d_sum = sum(sum(diagrk));
d_num = size(diagrk, 1) * size(diagrk, 1);
diagrk  = (d_sum/d_num - mean) / scaling;
