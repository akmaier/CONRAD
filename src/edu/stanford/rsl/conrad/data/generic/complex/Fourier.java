/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import ij.IJ;
import ij.ImageJ;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_3D;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class Fourier {

	public void fft(ComplexGrid input){
		fft(input,0);
	}

	public void fft(ComplexGrid input, int dim){
		if (input.getSize().length <= dim){
			throw new IllegalArgumentException("FFT along provided dimension not possible: Exceeds data dimension");
		}

		FloatFFT_1D fftInstance = new FloatFFT_1D(input.getSize()[dim]);
		// If dim == 0, the memory is in right shape and we can transform right away
		if (dim == 0){
			for (int i = 0; i < input.getNumberOfElements()/input.getSize()[dim]; i++) {
				fftInstance.complexForward(input.getAslinearMemory(), i*input.getSize()[dim]*2);
			}
			input.notifyAfterWrite();
		}
		else if(input.getSize().length==2){
			// Only FFT over columns are possible - Otherwise dim would be 0 and row-wise would be selected.
			for (int i = 0; i < input.getSize()[0]; i++){
				float[] cplxData = dataToStridedLine(input, new int[]{i,0}, dim);
				fftInstance.complexForward(cplxData);
				stridedLineToData(input, new int[]{i,0}, dim, cplxData);
			}
		}
		else if(input.getSize().length==3){
			// FFTs over columns and depth are possible - Rows would yield dim == 0 --> first case
			if (dim == 1){ // columns case
				for (int i = 0; i < input.getSize()[0]; i++){
					for (int k = 0; k < input.getSize()[2]; k++){
						float[] cplxData = dataToStridedLine(input, new int[]{i,0,k}, dim);
						fftInstance.complexForward(cplxData);
						stridedLineToData(input, new int[]{i,0,k}, dim, cplxData);
					}
				}
			}
			else{  // z-case
				for (int i = 0; i < input.getSize()[0]; i++){
					for (int j = 0; j < input.getSize()[1]; j++){
						float[] cplxData = dataToStridedLine(input, new int[]{i,j,0}, dim);
						fftInstance.complexForward(cplxData);
						stridedLineToData(input, new int[]{i,j,0}, dim, cplxData);
					}
				}
			}
		}
	}

	public void ifft(ComplexGrid input){
		ifft(input,0);
	}

	public void ifft(ComplexGrid input, int dim){
		if (input.getSize().length <= dim){
			throw new IllegalArgumentException("FFT along provided dimension not possible: Exceeds data dimension");
		}

		FloatFFT_1D fftInstance = new FloatFFT_1D(input.getSize()[dim]);
		// If dim == 0, the memory is in right shape and we can transform right away
		if (dim == 0){
			for (int i = 0; i < input.getNumberOfElements()/input.getSize()[dim]; i++) {
				fftInstance.complexInverse(input.getAslinearMemory(), i*input.getSize()[dim]*2,true);
			}
			input.notifyAfterWrite();
		}
		else if(input.getSize().length==2){
			// Only FFT over columns are possible - Otherwise dim would be 0 and row-wise would be selected.
			for (int i = 0; i < input.getSize()[0]; i++){
				float[] cplxData = dataToStridedLine(input, new int[]{i,0}, dim);
				fftInstance.complexInverse(cplxData,true);
				stridedLineToData(input, new int[]{i,0}, dim, cplxData);
			}
		}
		else if(input.getSize().length==3){
			// FFTs over columns and depth are possible - Rows would yield dim == 0 --> first case
			if (dim == 1){ // columns case
				for (int i = 0; i < input.getSize()[0]; i++){
					for (int k = 0; k < input.getSize()[2]; k++){
						float[] cplxData = dataToStridedLine(input, new int[]{i,0,k}, dim);
						fftInstance.complexInverse(cplxData,true);
						stridedLineToData(input, new int[]{i,0,k}, dim, cplxData);
					}
				}
			}
			else{  // z-case
				for (int i = 0; i < input.getSize()[0]; i++){
					for (int j = 0; j < input.getSize()[1]; j++){
						float[] cplxData = dataToStridedLine(input, new int[]{i,j,0}, dim);
						fftInstance.complexInverse(cplxData,true);
						stridedLineToData(input, new int[]{i,j,0}, dim, cplxData);
					}
				}
			}
		}
	}

	public void fft2(ComplexGrid input){
		fft2(input,0);
	}

	public void fft2(ComplexGrid input, int dim){
		if (input.getSize().length < 2){
			throw new IllegalArgumentException("FFT along provided dimension not possible: Exceeds data dimension");
		}

		if (input.getSize().length == 2){
			FloatFFT_2D fftInstance = new FloatFFT_2D(input.getSize()[1], input.getSize()[0]);
			fftInstance.complexForward(input.getAslinearMemory());
			input.notifyAfterWrite();
		}else{ // 3D case

			FloatFFT_2D fftInstance=null;
			switch(dim){
			case 0: // FFT along x/y planes
				fftInstance = new FloatFFT_2D(input.getSize()[1], input.getSize()[0]);
				for (int k = 0; k < input.getSize()[2]; k++) {
					float[] cplx = new float[input.getSize()[0]*input.getSize()[1]*2];
					System.arraycopy(input.getAslinearMemory(), k*cplx.length, cplx, 0, cplx.length);
					fftInstance.complexForward(cplx);
					System.arraycopy(cplx, 0, input.getAslinearMemory(), k*cplx.length, cplx.length);
				}
				input.notifyAfterWrite();
				break;
			case 1: // FFT along x/z planes
				fftInstance = new FloatFFT_2D(input.getSize()[2], input.getSize()[0]);
				for (int j = 0; j < input.getSize()[1]; j++) {
					float[] cplx = dataToPlane(input,j*input.getSize()[0],dim);
					fftInstance.complexForward(cplx);
					planeToData(input,j*input.getSize()[0],dim,cplx);
				}
				break;
			case 2: // FFT along y/z planes
				fftInstance = new FloatFFT_2D(input.getSize()[2], input.getSize()[1]);
				for (int i = 0; i < input.getSize()[0]; i++) {
					float[] cplx = dataToPlane(input,i,dim);
					//new ComplexGrid2D(cplx, 0, input.getSize()[1], input.getSize()[2]).show();
					fftInstance.complexForward(cplx);
					planeToData(input,i,dim,cplx);
				}
				break;
			default:
				break;
			}
		}
	}

	public void ifft2(ComplexGrid input){
		ifft2(input,0);
	}
	
	public void ifft2(ComplexGrid input, int dim){
		if (input.getSize().length < 2){
			throw new IllegalArgumentException("FFT along provided dimension not possible: Exceeds data dimension");
		}

		if (input.getSize().length == 2){
			FloatFFT_2D fftInstance = new FloatFFT_2D(input.getSize()[1], input.getSize()[0]);
			fftInstance.complexInverse(input.getAslinearMemory(),true);
			input.notifyAfterWrite();
		}else{ // 3D case

			FloatFFT_2D fftInstance=null;
			switch(dim){
			case 0: // FFT along x/y planes
				fftInstance = new FloatFFT_2D(input.getSize()[1], input.getSize()[0]);
				for (int k = 0; k < input.getSize()[2]; k++) {
					float[] cplx = new float[input.getSize()[0]*input.getSize()[1]*2];
					System.arraycopy(input.getAslinearMemory(), k*cplx.length, cplx, 0, cplx.length);
					fftInstance.complexInverse(cplx,true);
					System.arraycopy(cplx, 0, input.getAslinearMemory(), k*cplx.length, cplx.length);
				}
				input.notifyAfterWrite();
				break;
			case 1: // FFT along x/z planes
				fftInstance = new FloatFFT_2D(input.getSize()[2], input.getSize()[0]);
				for (int j = 0; j < input.getSize()[1]; j++) {
					float[] cplx = dataToPlane(input,j*input.getSize()[0],dim);
					fftInstance.complexInverse(cplx,true);
					planeToData(input,j*input.getSize()[0],dim,cplx);
				}
				break;
			case 2: // FFT along y/z planes
				fftInstance = new FloatFFT_2D(input.getSize()[2], input.getSize()[1]);
				for (int i = 0; i < input.getSize()[0]; i++) {
					float[] cplx = dataToPlane(input,i,dim);
					fftInstance.complexInverse(cplx,true);
					planeToData(input,i,dim,cplx);
				}
				break;
			default:
				break;
			}
		}
	}
	
	public void fft3(ComplexGrid input){
			FloatFFT_3D fftInstance = new FloatFFT_3D(input.getSize()[2], input.getSize()[1], input.getSize()[0]);
			fftInstance.complexForward(input.getAslinearMemory());
			input.notifyAfterWrite();
	}

	public void ifft3(ComplexGrid input){
			FloatFFT_3D fftInstance = new FloatFFT_3D(input.getSize()[2], input.getSize()[1], input.getSize()[0]);
			fftInstance.complexInverse(input.getAslinearMemory(),true);
			input.notifyAfterWrite();
	}
	
	

	private float[] dataToStridedLine(ComplexGrid grid, int[] idx, int dim){
		float[] out = new float[grid.getSize()[dim]*2];
		for (int i = 0; i < grid.getSize()[dim]; i++) {
			out[i*2]= (float)grid.getValue(idx).getReal();
			out[i*2+1]=(float)grid.getValue(idx).getImag();
			idx[dim]++;
		}
		return out;
	}

	private void stridedLineToData(ComplexGrid grid,  int[] idx, int dim, float[] fftBuffer){
		for (int i = 0; i < grid.getSize()[dim]; i++) {
			grid.setValue(new Complex(fftBuffer[i*2],fftBuffer[i*2+1]),idx);
			idx[dim]++;
		}
	}

	private float[] dataToPlane(ComplexGrid grid, int startPos, int dim){
		float[] out = null;
		switch(dim){
		case 1: //over x/z planes
			out = new float[grid.getSize()[0]*grid.getSize()[2]*2];
			for (int k = 0; k < grid.getSize()[2]; k++) {
				System.arraycopy(grid.getAslinearMemory(), (k*grid.getSize()[0]*grid.getSize()[1]+startPos)*2, out, k*grid.getSize()[0]*2, grid.getSize()[0]*2);
			}
			break;
		case 2: //over y/z planes
			out = new float[grid.getSize()[1]*grid.getSize()[2]*2];
			for (int k = 0; k < grid.getSize()[2]; k++) {
				for (int j = 0; j < grid.getSize()[1]; j++) {
					out[(k*grid.getSize()[1]+j)*2] = grid.getAslinearMemory()[2*(startPos+j*grid.getSize()[0]+k*grid.getSize()[0]*grid.getSize()[1])];
					out[(k*grid.getSize()[1]+j)*2+1] = grid.getAslinearMemory()[2*(startPos+j*grid.getSize()[0]+k*grid.getSize()[0]*grid.getSize()[1])+1];
				}
			}
			break;
		default:
			break;
		}
		return out;
	}

	private void planeToData(ComplexGrid grid, int startPos, int dim, float[] cplx){
		switch(dim){
		case 1: //over x/z planes
			for (int k = 0; k < grid.getSize()[2]; k++) {
				//System.arraycopy(grid.getAslinearMemory(), (k*grid.getSize()[0]*grid.getSize()[1]+startPos)*2, out, k*grid.getSize()[0], grid.getSize()[0]);
				System.arraycopy(cplx, k*grid.getSize()[0]*2, grid.getAslinearMemory(), (k*grid.getSize()[0]*grid.getSize()[1]+startPos)*2, grid.getSize()[0]*2);
			}
			grid.notifyAfterWrite();
			break;
		case 2: //over y/z planes
			for (int k = 0; k < grid.getSize()[2]; k++) {
				for (int j = 0; j < grid.getSize()[1]; j++) {
					grid.getAslinearMemory()[2*(startPos+j*grid.getSize()[0]+k*grid.getSize()[0]*grid.getSize()[1])]=cplx[(k*grid.getSize()[1]+j)*2];
					grid.getAslinearMemory()[2*(startPos+j*grid.getSize()[0]+k*grid.getSize()[0]*grid.getSize()[1])+1]=cplx[(k*grid.getSize()[1]+j)*2+1];
				}
			}
			grid.notifyAfterWrite();
			break;
		default:
			break;
		}
	}

	public static void main(String[] args) {
		new ImageJ();
		Grid3D blubb = ImageUtil.wrapImagePlus(IJ.openImage("D:\\Data\\ClackdoylePhantom\\Clackdoyle3D.tif"));
		/*Grid3D blubb = new Grid3D(4,4,4);
		DoubleFunction fct = (x -> Math.abs((double)x-1.5));
		for (int i = 0; i < blubb.getSize()[0]; i++) {
			for (int j = 0; j < blubb.getSize()[1]; j++) {
				for (int k = 0; k < blubb.getSize()[2]; k++) {
					if (fct.f(i) < 1 && fct.f(j) < 1 && fct.f(k) < 1)
						blubb.setAtIndex(i, j, k, 1);
				}
			}
		}*/

		Fourier ft = new Fourier();

		
		ComplexGrid cg3d = new ComplexGrid3D(blubb);
		ComplexGrid cg3dRef = (ComplexGrid3D)cg3d.clone();

		ComplexGrid cg2d = ((ComplexGrid3D)cg3d).getSubGrid(300).clone();
		ComplexGrid cg2dRef = (ComplexGrid2D)cg2d.clone();

		
		ComplexGrid cg1d = ((ComplexGrid2D)cg2d).getSubGrid(300).clone();
		ComplexGrid cg1dRef = (ComplexGrid1D)cg1d.clone();

		// 1D FFT testing
		ft.fft(cg1d);ft.ifft(cg1d);
		ft.fft(cg2d,0);ft.ifft(cg2d,0);ft.fft(cg2d,1);ft.ifft(cg2d,1);
		ft.fft(cg3d,0);ft.ifft(cg3d,0);ft.fft(cg3d,1);ft.ifft(cg3d,1);ft.fft(cg3d,2);ft.ifft(cg3d,2);
		


		ft.fft2(cg2d);ft.ifft2(cg2d);
		ft.fft2(cg3d,0);ft.ifft2(cg3d,0);ft.fft2(cg3d,1);ft.ifft2(cg3d,1);ft.fft2(cg3d,2);ft.ifft2(cg3d,2);


		ft.fft3(cg3d);
		ft.ifft3(cg3d);
		 
		ComplexGridOperator cgo = new ComplexGridOperator();
		cgo.subtractBy(cg1dRef, cg1d);
		cg1dRef.show("1D-Diff");
		cgo.subtractBy(cg2dRef, cg2d);
		cg2dRef.show("2D-Diff");
		cgo.subtractBy(cg3dRef, cg3d);
		cg3dRef.show("3D-Diff");
		

	}
}

