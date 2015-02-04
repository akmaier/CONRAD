clear variables
close all
clc

import ij.*
import edu.stanford.rsl.conrad.data.generic.complex.*
import edu.stanford.rsl.conrad.data.generic.*
import edu.stanford.rsl.conrad.utils.*


try
    jImg3D = loadImageJVolume('D:\Data\ClackdoylePhantom\Clackdoyle3D.tif');
    mImg3D = single(grid3D2mat(jImg3D));
    
    asiz = size(mImg3D);
    
    jImg2D = jImg3D.getSubGrid(300).clone();
    mImg2D = reshape(jImg2D.getBuffer(),jImg2D.getWidth(),jImg2D.getHeight());
    
    jImg1D = jImg2D.getSubGrid(300).clone();
    mImg1D = jImg1D.getBuffer();
    
    jImg3D = ComplexGrid3D(jImg3D);
    jImg2D = ComplexGrid2D(jImg2D);
    jImg1D = ComplexGrid1D(jImg1D);
    
    ft = Fourier();
    
    j1DtoMat = @(x) (1i*x().getImagGrid().getBuffer() + x().getRealGrid().getBuffer());
    j2DtoMat = @(x) (reshape(1i*x().getImagGrid().getBuffer(),asiz(1)*asiz(2),1) + reshape(x().getRealGrid().getBuffer(),asiz(1)*asiz(2),1));
    j3DtoMat = @(x) (reshape(1i*grid3D2mat(x().getImagGrid()),prod(asiz),1) + reshape(grid3D2mat(x().getRealGrid()),prod(asiz),1));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1D Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %1D FFTs on 1D line
    jImg1DClone = jImg1D.clone();
    mImg1DClone = mImg1D;
    ft.fft(jImg1DClone);
    mImg1DClone=fft(mImg1DClone);
    errMSE(1,1) = norm(abs(j1DtoMat(jImg1DClone)-mImg1DClone)).^2/asiz(1);
    errMax(1,1) = max(abs(j1DtoMat(jImg1DClone)-mImg1DClone(:)));
    ft.ifft(jImg1DClone);
    mImg1DClone=ifft(mImg1DClone);
    errMSE(1,2) = norm(abs(j1DtoMat(jImg1DClone)-mImg1DClone)).^2/asiz(1);
    errMax(1,2) = max(abs(j1DtoMat(jImg1DClone)-mImg1DClone(:)));
    
    %1D FFTs on 2D along x
    jImg2DClone = jImg2D.clone();
    mImg2DClone = mImg2D;
    ft.fft(jImg2DClone,0);
    mImg2DClone = fft(mImg2DClone);
    errMSE(end+1,1) = norm(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:))).^2/prod(asiz(1:2));
    errMax(end+1,1) = max(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:)));
    ft.ifft(jImg2DClone,0);
    mImg2DClone = ifft(mImg2DClone);
    errMSE(end,2) = norm(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:))).^2/prod(asiz(1:2));
    errMax(end,2) = max(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:)));
    
    %1D FFTs on 2D along y
    jImg2DClone = jImg2D.clone();
    mImg2DClone = mImg2D;
    ft.fft(jImg2DClone,1);
    mImg2DClone = fft(mImg2DClone,[],2);
    errMSE(end+1,1) = norm(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:))).^2/prod(asiz(1:2));
    errMax(end+1,1) = max(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:)));
    ft.ifft(jImg2DClone,1);
    mImg2DClone = ifft(mImg2DClone,[],2);
    errMSE(end,2) = norm(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:))).^2/prod(asiz(1:2));
    errMax(end,2) = max(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:)));
    
    
    %1D FFTs on 3D along x
    %bla = abs(j3DtoMat(jImg3DClone)-mImg3DClone(:));
    %picker = randperm(length(bla));
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    fdim = 0;
    ft.fft(jImg3DClone,fdim);
    mImg3DClone = fft(mImg3DClone,[],fdim+1);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft(jImg3DClone,fdim);
    mImg3DClone = ifft(mImg3DClone,[],fdim+1);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    fdim = 1;
    ft.fft(jImg3DClone,fdim);
    mImg3DClone = fft(mImg3DClone,[],fdim+1);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft(jImg3DClone,fdim);
    mImg3DClone = ifft(mImg3DClone,[],fdim+1);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    fdim = 2;
    ft.fft(jImg3DClone,fdim);
    mImg3DClone = fft(mImg3DClone,[],fdim+1);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft(jImg3DClone,fdim);
    mImg3DClone = ifft(mImg3DClone,[],fdim+1);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %2D Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %2D image on 2D FFT
    jImg2DClone = jImg2D.clone();
    mImg2DClone = mImg2D;
    ft.fft2(jImg2DClone,0);
    mImg2DClone = fft2(mImg2DClone);
    errMSE(end+1,1) = norm(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:))).^2/prod(asiz(1:2));
    errMax(end+1,1) = max(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:)));
    ft.ifft2(jImg2DClone,0);
    mImg2DClone = ifft2(mImg2DClone);
    errMSE(end,2) = norm(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:))).^2/prod(asiz(1:2));
    errMax(end,2) = max(abs(j2DtoMat(jImg2DClone)-mImg2DClone(:)));
    
    %2D FFTs on 3D along xy planes
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    fdim = 0;
    ft.fft2(jImg3DClone,fdim);
    mImg3DClone = fft2(mImg3DClone);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft2(jImg3DClone,fdim);
    mImg3DClone = ifft2(mImg3DClone);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
    %2D FFTs on 3D along xz planes
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    fdim = 1;
    ft.fft2(jImg3DClone,fdim);
    mImg3DClone = fft(fft(mImg3DClone,[],1),[],3);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft2(jImg3DClone,fdim);
    mImg3DClone = ifft(ifft(mImg3DClone,[],1),[],3);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
    %2D FFTs on 3D along yz planes
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    fdim = 2;
    ft.fft2(jImg3DClone,fdim);
    mImg3DClone = fft(fft(mImg3DClone,[],2),[],3);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft2(jImg3DClone,fdim);
    mImg3DClone = ifft(ifft(mImg3DClone,[],2),[],3);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %3D Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    jImg3DClone = jImg3D.clone();
    mImg3DClone = mImg3D;
    ft.fft3(jImg3DClone);
    mImg3DClone = fftn(mImg3DClone);
    errMSE(end+1,1) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end+1,1) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    ft.ifft3(jImg3DClone);
    mImg3DClone = ifftn(mImg3DClone);
    errMSE(end,2) = norm(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:))).^2/prod(asiz);
    errMax(end,2) = max(abs(j3DtoMat(jImg3DClone)-mImg3DClone(:)));
    
catch errCatch
    error(errCatch.stack)
end