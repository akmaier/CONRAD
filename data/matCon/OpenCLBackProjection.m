function[volume] = OpenCLBackProjection(projectionsIn, backProjector)
import edu.stanford.rsl.conrad.data.numeric.*
import edu.stanford.rsl.conrad.utils.*
import edu.stanford.rsl.conrad.opencl.*

% check if input is a java volume, if not make it a java volume
if(~isa(projectionsIn,'Grid3D'))
    assert(length(size(projectionsIn))==3)
    projectionsIn = mat2Grid3D(projectionsIn);
end

try
    % no forward projector given, create new one
    if (nargin<2 || isempty(backProjector))
        backProjector = OpenCLBackProjector();
    end
    backProjector.configure();
    filteringArray=javaArray('edu.stanford.rsl.conrad.filtering.ImageFilteringTool',1);
    filteringArray(1)=backProjector;
    volume = ImageUtil.applyFiltersInParallel(projectionsIn,filteringArray);
catch err
    disp(err.message);
end

end