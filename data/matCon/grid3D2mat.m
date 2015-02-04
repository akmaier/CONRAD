function[matOut] = grid3D2mat(inputVolume)
dimen = inputVolume.getSize();
matOut = zeros(dimen.');
for t=1:dimen(3)
    slice = inputVolume.getSubGrid(t-1).getBuffer();
    if isa(inputVolume,'edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D')
        slice = (slice(mod(1:length(slice),2)==1)+1i*slice(mod(1:length(slice),2)==0));
    end
    matOut(:,:,t)= reshape(slice,dimen(1),dimen(2));
end