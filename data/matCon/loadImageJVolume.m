function[volume]=loadImageJVolume(filename)

import ij.*
import edu.stanford.rsl.conrad.data.numeric.*
import edu.stanford.rsl.conrad.utils.*
try
    if (isempty(filename))
        filename = FileUtil.myFileChoose('',false);
    end
    ijvolume = IJ.openImage(filename);
    volume = ImageUtil.wrapImagePlus(ijvolume);
catch exception
    disp(exception.message);
    exception.stack
    volume=[];
end

end