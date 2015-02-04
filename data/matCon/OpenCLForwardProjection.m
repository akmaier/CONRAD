function[projections] = OpenCLForwardProjection(volume, forwardProjector)
import edu.stanford.rsl.conrad.data.numeric.*
import edu.stanford.rsl.conrad.utils.*
import edu.stanford.rsl.conrad.opencl.*



% check if input is a java volume, if not make it a java volume
if(~isa(volume,'Grid3D'))
    assert(length(size(volume))==3)
    volume = mat2Grid3D(volume);
end

try
    % no forward projector given, create new one
    if (nargin < 2 || isempty(forwardProjector))
        forwardProjector = OpenCLForwardProjector();
    end
    forwardProjector.configure();
    forwardProjector.setTex3D(ImageUtil.wrapGrid3D(volume,'ProjectionVolume'));
    projections = ImageUtil.wrapImagePlus(forwardProjector.project());
catch err
    disp(err.message);
end
end