clear variables
close all
clc

% Example for the forward and backward projection using CONRAD
% (1) import java packages
import ij.*
import edu.stanford.rsl.conrad.utils.*
import edu.stanford.rsl.conrad.data.numeric.*

% (2) Load the CONRAD settings xml file
try
    % default config file as provided 
    % (just replace path to your own config file if needed)
    config = Configuration.loadConfiguration(fullfile(fileparts(mfilename('fullpath')),'Conrad.xml'));
    Configuration.setGlobalConfiguration(config);
catch err
    error('Could not load valid CONRAD config file!');
end

% (3) open ImageJ / if already open, close all images
if (isempty(IJ.getInstance()))
    ImageJ();
else
    while(WindowManager.getImageCount()>0)
        openImg = IJ.getImage();
        openImg.close();
    end
end


% (4) define a matlab volume (two spheres with high and low density)
volDim = [512,512,512];
[x,y,z]=meshgrid(1:volDim(1),1:volDim(2),1:volDim(3));
vol=zeros(volDim);
% sphere 1
vol(sqrt((x-(volDim(1)+1)/2).^2+(y-(volDim(2)+1)/2).^2+(z-(volDim(3)+1)/2).^2) < min(volDim)*0.5/2)=1;
% sphere 2
vol(sqrt((x-(volDim(1)+1)/2).^2+(y-(volDim(2)+1)/2).^2+(z-(volDim(3)+1)/2).^2) < min(volDim)*0.3/2)=4;

% (5) OpenCL forward projection
projections=OpenCLForwardProjection(vol);
projections.show('Forward Projected Matlab Object');

% (6) OpenCL backprojection without filtering
volRec1=OpenCLBackProjection(projections);
volRec1.show('Backprojection without filtering');

% (7) A full reconstruction with filtering
volRec2=reconstructionHelper(projections);
volRec2.show('Full reconstruction with filtering');

% (8) Show also the Ground Truth in ImageJ
gT = mat2Grid3D(vol);
gT.show('Ground Truth');

% If the outcome of the forward or backward projections are needed in
% Matlab format use the following (This copies the whole volume --> slow):
% mProjections = grid3D2mat(projections);

% However, you can also access the Grid3D directly (much faster) by using
% for example "projections.setAtIndex(x,y,z,value)".