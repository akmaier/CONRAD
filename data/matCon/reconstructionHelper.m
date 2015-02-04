function[volume, backProjector] = reconstructionHelper(projectionsIn,motion,doShortScan)
if ( nargin < 2)
    motion=[];
end
if (nargin < 3)
    doShortScan = false;
end
import edu.stanford.rsl.conrad.data.numeric.*
import edu.stanford.rsl.conrad.utils.*
import edu.stanford.rsl.conrad.opencl.*
import edu.stanford.rsl.conrad.filtering.*
import edu.stanford.rsl.conrad.filtering.rampfilters.*
import edu.stanford.rsl.conrad.filtering.redundancy.*
import edu.stanford.rsl.conrad.reconstruction.*

% check if input is a java volume, if not make it a java volume
if(~isa(projectionsIn,'Grid3D'))
    assert(length(size(projectionsIn))==3)
    projectionsIn = mat2Grid3D(projectionsIn);
end

if (doShortScan)
    parker = ParkerWeightingTool();
end
cosine = CosineWeightingTool();
kernel = SheppLoganRampFilter();
kernel.setConfiguration(Configuration.getGlobalConfiguration());

try
    kernel.configure();
catch err
    disp(err.message);
end

if (~isempty(motion))
    backProjector = OpenCLDetectorMotionBackProjector();
else
    backProjector = OpenCLBackProjector();
end
filter = RampFilteringTool();
filter.setRamp(kernel);
filter.setConfigured(true);

try
    if (doShortScan)
        parker.configure();
    end
    cosine.configure();
    if (~isempty(motion))
        backProjector.configure(motion);
    else
        backProjector.configure();
    end
    
catch err
    disp(err.message);
end

if (doShortScan)
    filteringArray=javaArray('edu.stanford.rsl.conrad.filtering.ImageFilteringTool',4);
    filteringArray(1)=cosine;
    filteringArray(2)=parker;
    filteringArray(3)=filter;
    filteringArray(4)=backProjector;
else
    filteringArray=javaArray('edu.stanford.rsl.conrad.filtering.ImageFilteringTool',3);
    filteringArray(1)=cosine;
    filteringArray(2)=filter;
    filteringArray(3)=backProjector;
end


volume = ImageUtil.applyFiltersInParallel(projectionsIn,filteringArray);
end