function[grid3Dout] = mat2Grid3D(inputVolume)
import edu.stanford.rsl.conrad.data.numeric.*
import edu.stanford.rsl.conrad.data.generic.complex.*

if isreal(inputVolume)
    grid3Dout = Grid3D(size(inputVolume,1),size(inputVolume,2),size(inputVolume,3),false);
else
    grid3Dout = ComplexGrid3D(size(inputVolume,1),size(inputVolume,2),size(inputVolume,3));
end
for t=1:size(inputVolume,3)
    slice = inputVolume(:,:,t);
    if isreal(inputVolume)
        g2d = Grid2D(slice(:),size(inputVolume,1),size(inputVolume,2));
    else
        grid3Dout = ComplexGrid3D(size(inputVolume,1),size(inputVolume,2),size(inputVolume,3));
        for i=1:size(inputVolume,1)
            for j=1:size(inputVolume,2)
                for k=1:size(inputVolume,3)
                    grid3Dout.setRealAtIndex(real(inputVolume(i,j,k)),[i,j,k]-1);
                    grid3Dout.setImagAtIndex(imag(inputVolume(i,j,k)),[i,j,k]-1);
                end
            end
        end
        break;
    end
    
    grid3Dout.setSubGrid(t-1,g2d);
end
