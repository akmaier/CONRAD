% This function releases all gpu resources allocate by the static context
% inside Conrad's OpenCLUtil class
function[] = conradCleanUp()
import edu.stanford.rsl.conrad.opencl.*
OpenCLUtil.releaseStaticContext();
end