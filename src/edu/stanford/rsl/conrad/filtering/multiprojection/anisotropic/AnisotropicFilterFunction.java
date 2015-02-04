package edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic;

import ij.IJ;


import java.util.Arrays;

import edu.stanford.rsl.conrad.cuda.CUDAVolume3D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.volume3d.FFTVolumeHandle;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import edu.stanford.rsl.conrad.volume3d.VolumeOperator;
import edu.stanford.rsl.conrad.volume3d.VolumeOperator.FILTER_TYPE;

/**
 * Class to process a Volume3D with an anisotropic structure tensor filter.
 * 
 * @author akmaier
 *
 */
public class AnisotropicFilterFunction {

	protected boolean showStatus = true;
	protected boolean debug = false;

	public void setShowStatus (boolean showStatus){
		this.showStatus = showStatus;
	}

	private FFTVolumeHandle fft;
	private VolumeOperator operator;

	private final static boolean DEBUG_FLAG = false;
	public final static int MAX_FILTERS = 12;

	public AnisotropicFilterFunction (FFTVolumeHandle fft, VolumeOperator op){
		this.fft = fft;
		operator = op;
	}

	private static void printf(Object s){
		System.out.print(s);
	}

	/**
	 * Method to limit the number of threads used by the FFT of the filter.
	 * @param number
	 */
	public void setThreadNumber(int number){
		fft.setThreadNumber(number);
	}
	
	private static float sqrt(double in){
		return (float) Math.sqrt(in);
	}	

	float SIGN(float x) {
		return ((x) >= 0 ?  (float) 1 : (float) -1);
	}

	public void filt_gauss_relax_tensors(Volume3D [][] vol, float a)
	{
		Volume3D filt;

		filt = operator.createGaussLowPassFilter(vol[0][0].dimensions, vol[0][0].size, vol[0][0].spacing, a);
		int senbu = vol[0][0].dimensions * (vol[0][0].dimensions+1) /2;
		int index = 0;
		for (int i =0; i<vol.length; i++) {
			for (int j =i; j<vol[i].length; j++) {
				if (showStatus) {
					IJ.showStatus("Relaxing Structure Tensor");
					IJ.showProgress((((float)(index))/senbu));
				}
				index ++;
				fft.forwardTransform(vol[i][j]);

				operator.multiply(vol[i][j], filt);

				fft.inverseTransform(vol[i][j]);

				operator.real(vol[i][j]);    /*  OBS !!!  a real and even . vol ~ real */

				//if (debug) if (vol[i][j] instanceof CUDAVolume3D){
				//	((CUDAVolume3D) vol[i][j]).fetch();
				//	vol[i][j].getImagePlus("relaxed tensor " + i +"  " + j).show();
				//}
				CONRAD.gc();
			}

		}
		filt.destroy();
	}

	public Volume3D [][] computeStructureTensor(Volume3D vol, int A, float B, float ri)
	{
		Volume3D vol1, vol2, filt [];
		float [] [] dir = new float [MAX_FILTERS][Volume3D.MAX_DIM];
		float [] [] t_coeff = new float [Volume3D.MAX_DIM][Volume3D.MAX_DIM];
		float [] dim = new float[Volume3D.MAX_DIM];
		int    filt_loop, row, col, dimensions, n_filters;
		int [] size = new int [Volume3D.MAX_DIM];

		/* filter and compute the sqare magnutude in each direction */

		//Volume3D.vol_fft(vol);
		//if (debug) if (vol instanceof CUDAVolume3D){
		//	((CUDAVolume3D) vol).fetch();
		//	vol.getImagePlus("FFT Vol").show();
		//}

		dimensions = vol.dimensions;
		filt_get_filt_dirs(vol.dimensions, dir);
		n_filters = filt_get_n_filters(vol.dimensions);
		filt = new Volume3D[n_filters];
		for (filt_loop=0; filt_loop<n_filters; filt_loop++) {
			if (showStatus) {
				IJ.showStatus("Preparing Quadrature Filteres");
				IJ.showProgress((((float)(filt_loop))/n_filters));
			}
			filt[filt_loop] = operator.createExponentialDirectionalHighPassFilter(vol.dimensions, vol.size, vol.spacing,
					dir[filt_loop], A, B, ri, FILTER_TYPE.QUADRATIC);

			if (filt[filt_loop]==null) {
				printf( "filt_orient: Error creating filter\n");
				return(null);
			}

			operator.multiply(filt[filt_loop], vol);

			fft.inverseTransform(filt[filt_loop]);
			operator.abs(filt[filt_loop]);
			CONRAD.gc();
		}



		/* Get size and spacing from the original (complex) volume */

		size = vol.size;
		dim = vol.spacing;
		Volume3D [] [] tensor = new Volume3D[vol.dimensions][vol.dimensions];

		int senbu = dimensions * (dimensions+1) /2;
		int index = 0;
		for (row=0; row<dimensions; row++){
			for (col=row; col<dimensions; col++) {
				if (showStatus) {
					IJ.showStatus("Assembling Structure Tensor");
					IJ.showProgress((((float)(index))/senbu));
				}
				index ++;
				/* Create a new non-complex volume */

				vol1=operator.createVolume(size, dim, 1);

				/* Add each filter response multiplied by the corresponding constant */

				for (filt_loop=0; filt_loop<n_filters; filt_loop++) {

					vol2 = filt[filt_loop];

					filt_calc_Nk_I(dimensions, filt_loop, t_coeff);

					double weight = t_coeff[row][col];

					if (row != col) weight*=2;          //  Removed LW 990320, added again 060208

					operator.addVolume(vol1, vol2, weight);
					vol2 = null;
					if((row==col)&&(col==dimensions-1)){
						filt[filt_loop].destroy();
						filt[filt_loop] = null;
					}
				}
				tensor[row][col] =vol1;
				if (debug) if (tensor[row][col] instanceof CUDAVolume3D){
					((CUDAVolume3D) tensor[row][col]).fetch();
					tensor[row][col].getImagePlus("Tensor" + row + " " + col).show();
				}
				vol1 = null;
			}
		}
		filt = null;
		return tensor;
	}



	public static int filt_get_n_filters(int dimensions)
	{
		switch (dimensions) {

		case 2:
			return(3);
		case 3:
			return(6);
		case 4:
			return(12);
		default:
			printf("filt_get_n_filters: Invalid dimension\n");
			return(-1);

		}
	}

	/**
	 * Computes the filter directions. Result is written into @param fd
	 * @param dimensions the number of dimensions
	 * @param fd the result
	 * @return a return variable
	 */
	public static int filt_get_filt_dirs(int dimensions, float [][] fd)
	{
		float a,b,c;

		for (int i =0; i< fd.length; i++)
			Arrays.fill(fd[i], 0);

		switch (dimensions) {

		case 2:

			fd[0][0] =  1.0f;  fd[0][1] = 0;
			fd[1][0] =  0.5f;  fd[1][1] = (float) (sqrt(3.0)/2.0);
			fd[2][0] = -0.5f;  fd[2][1] = (float) (sqrt(3.0)/2.0);

			break;

		case 3:

			a = (float) (2.0/sqrt(10.0+2.0*sqrt(5.0)));
			b = (float) ((1.0+sqrt(5.0))/sqrt(10.0+2.0*sqrt(5.0)));

			fd[0][0] =  a;  fd[0][1] =  0;  fd[0][2] =  b;
			fd[1][0] = -a;  fd[1][1] =  0;  fd[1][2] =  b;
			fd[2][0] =  b;  fd[2][1] =  a;  fd[2][2] =  0;
			fd[3][0] =  b;  fd[3][1] = -a;  fd[3][2] =  0;
			fd[4][0] =  0;  fd[4][1] =  b;  fd[4][2] =  a;
			fd[5][0] =  0;  fd[5][1] =  b;  fd[5][2] = -a;

			break;

		case 4:

			c = (float) (1.0/sqrt(2.0));

			fd[ 0][0] =  c;  fd[ 0][1] =  c;  fd[ 0][2] =  0;  fd[ 0][3] =  0;
			fd[ 1][0] =  c;  fd[ 1][1] = -c;  fd[ 1][2] =  0;  fd[ 1][3] =  0;
			fd[ 2][0] =  c;  fd[ 2][1] =  0;  fd[ 2][2] =  c;  fd[ 2][3] =  0;
			fd[ 3][0] =  c;  fd[ 3][1] =  0;  fd[ 3][2] = -c;  fd[ 3][3] =  0;
			fd[ 4][0] =  c;  fd[ 4][1] =  0;  fd[ 4][2] =  0;  fd[ 4][3] =  c;
			fd[ 5][0] =  c;  fd[ 5][1] =  0;  fd[ 5][2] =  0;  fd[ 5][3] = -c;
			fd[ 6][0] =  0;  fd[ 6][1] =  c;  fd[ 6][2] =  c;  fd[ 6][3] =  0;
			fd[ 7][0] =  0;  fd[ 7][1] =  c;  fd[ 7][2] = -c;  fd[ 7][3] =  0;
			fd[ 8][0] =  0;  fd[ 8][1] =  c;  fd[ 8][2] =  0;  fd[ 8][3] =  c;
			fd[ 9][0] =  0;  fd[ 9][1] =  c;  fd[ 9][2] =  0;  fd[ 9][3] = -c;
			fd[10][0] =  0;  fd[10][1] =  0;  fd[10][2] =  c;  fd[10][3] =  c;
			fd[11][0] =  0;  fd[11][1] =  0;  fd[11][2] =  c;  fd[11][3] = -c;

			break;

		default:

			printf("filt_get_filt_dirs: Invalid dimension ("+dimensions+")\n");
			return(-1);

		}

		return(0);
	}


	/**
	 * Computes the orientation tensor {@latex.inline $\\mathbf{M}_k = l\\mathbf{\\hat{n}}_k\\mathbf{\\hat{n}}_k^\\top-m\\mathbf{I}$}, where k is the index of the filter direction,
	 * {@latex.inline $\\mathbf{I}$} is the identity matrix, {@latex.inline $\\mathbf{\\hat{n}}_k$} the orientation of the filter. l is 4/3 for 2D and 5/4 for 3D, m is 1/3 for 2D and 1/4 for 3D.
	 * @param dimensions the filter dimension
	 * @param k the filter index
	 * @param t_coeff the orientation tensor. 
	 * @return 0
	 */
	public static int filt_calc_Nk_I(int dimensions,
			int k, float [] []t_coeff)
	{
		float [][] fd = new float[MAX_FILTERS][Volume3D.MAX_DIM];
		int   dim_loop, row, col;
		float c_mul,c_sub;

		filt_get_filt_dirs(dimensions, fd);

		/*             t    */
		/*    N  = n  n     */
		/*     k    k  k    */

		switch (dimensions) {

		case 2:
			c_mul = (float) (4.0/3.0);
			c_sub = (float) (1.0/3.0);
			break;
		case 3:
			c_mul = (float) (5.0/4.0);
			c_sub = (float) (1.0/4.0);
			break;
		case 4:
			c_mul = (float) 1.0;
			c_sub = (float) (1.0/6.0);
			break;
		default:
			printf("filt_calc_Nk_I: Invalid dimension ("+dimensions+")\n");
			return(-1);
		}

		for (row=0; row<dimensions; row++)
			for (col=0; col<dimensions; col++)
				t_coeff[row][col] = c_mul * fd[k][row] * fd[k][col];

		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			t_coeff[dim_loop][dim_loop] -= c_sub;

		return 0;
	}



	@Deprecated
	protected Volume3D [] filt_make_enhance_filters(int dimensions,
			int size[], float dim[], float lpUpper)
	{

		Volume3D [] filters;
		int n_filters = filt_get_n_filters(dimensions);


		if (DEBUG_FLAG)
			printf("filt_make_enhance_filters\n");


		filters = new Volume3D[n_filters+1];
		filters[0] = operator.createLowPassFilter(dimensions, size, dim, lpUpper);

		for (int filt_loop=0; filt_loop<n_filters; filt_loop++) {
			filters[filt_loop+1] = operator.createHighPassFilter(dimensions, size, dim, filt_loop, lpUpper);
		}

		return(filters);
	}

	
	/**
	 * Normalizes the tensors and computes the tensor nor. Note that the norm is computed BEFORE the normalization.
	 * Computes {@latex.inline $\\mathbf{\\hat{T}} = \\frac{\\mathbf{T}}{{\\lambda_1}}$} with {@latex.inline ${\\lambda_1}$} begin the normalization factor. It is the either the trace or the largest eigenvalue of the vector. 
	 * Furthermore, the method returns the L2-norm of the tensor {@latex.inline $|\\mathbf{T}|$}
	 * 
	 * @param structureTensor the reference to the structure tensor
	 * @param eig if eig == 1 the maximal eigenvalue is used for normalization else the norm is performed using the tensor trace.
	 * @return the tensor norm before normalization. 
	 */
	public Volume3D filt_normalize_tensors(Volume3D [][] structureTensor, int eig)
	{
		Volume3D vol, vol2;
		int row, col;
		int dim_loop;
		float [] dim = new float [Volume3D.MAX_DIM];
		int dimensions;
		int [] size = new int [Volume3D.MAX_DIM];

		/* Get size and spacing */


		dimensions = structureTensor[0][0].dimensions;
		for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
			size[dim_loop] = structureTensor[0][0].size[dim_loop];
			dim[dim_loop] = structureTensor[0][0].spacing[dim_loop];
		}

		for (row=0; row<dimensions; row++) {
			for (col=row; col<dimensions; col++) {
				if (debug) if (structureTensor[row][col] instanceof CUDAVolume3D){
					((CUDAVolume3D) structureTensor[row][col]).fetch();
					structureTensor[row][col].getImagePlus("st " + row + " " + col).show();
				}
			}
		}

		if (eig == 1) {

			/* Max eigenvalue */

			vol=operator.solveMaximumEigenvalue(structureTensor);

		} else {

			/* Trace */

			vol=operator.createVolume(size, dim, 1);
			for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
				vol2=structureTensor[dim_loop][dim_loop];
				operator.addVolume(vol,vol2);
			}
		}

		if (debug) if (vol instanceof CUDAVolume3D){
			((CUDAVolume3D) vol).fetch();
			vol.getImagePlus("Eigenvalues").show();
		}

		/* calculate ABSfrob(T) */

		Volume3D norm=operator.createVolume(size, dim, 1);

		for (row=0; row<dimensions; row++){
			for (col=row; col<dimensions; col++) {

				vol2=operator.createVolume(structureTensor[row][col].size, structureTensor[row][col].spacing, structureTensor[row][col].getInternalDimension());
				operator.addVolume(vol2, structureTensor[row][col]);
				operator.multiply(vol2,vol2);
				double weight = 1;
				if (row != col) weight = 2.0;
				operator.addVolume(norm,vol2,weight);
				vol2.destroy();
			}
		}
		operator.abs(norm);
		operator.sqrt(norm);
		
		/* calculate T^ */

		for (row=0; row<dimensions; row++) {
			for (col=row; col<dimensions; col++) {

				vol2=structureTensor[row][col];
				operator.divideByVolume(vol2,vol);

				if (debug) if (vol2 instanceof CUDAVolume3D){
					((CUDAVolume3D) vol2).fetch();
					vol2.getImagePlus("Normalized by Eigenvalue " + row + " " + col).show();
				}

			}
		}
		vol.destroy();
		
		



		
		
	

		return norm;
	}

	/**
	 * Computes an array of filtered images. The entry at [0] contains the low pass filtered image.
	 * The entry at [1] contains the largest eigenvalues of the structure tensors computed at each voxel of the volume. Enrty [2] contains the directional high pass filtered image.
	 * @param volume the volume to process
	 * @param A
	 * @param B
	 * @param ri
	 * @param a
	 * @param lpUpper low pass frequency.
	 * @return the array of volumes.
	 */
	public Volume3D [] filt_pre_enhance(Volume3D volume, int A, float B, float ri, float a, float lpUpper)
	{
		Volume3D vol, vol2, filt;
		int loop, row, col;
		int  dim_loop;
		float [][] t_coeff = new float [Volume3D.MAX_DIM][Volume3D.MAX_DIM];
		float [] dim = new float[Volume3D.MAX_DIM];
		int dimensions;
		int [] size = new int [Volume3D.MAX_DIM];
		int n_filters;


		if (debug) if (volume instanceof CUDAVolume3D){
			((CUDAVolume3D) volume).fetch();
			volume.getImagePlus("Vol").show();
		}

		fft.forwardTransform(volume);
		Volume3D [] revan = new Volume3D[3];


		if(debug) System.out.println("creating tensor");
		Volume3D [][] tensor = computeStructureTensor(volume, A, B, ri);
		CONRAD.gc();
		if(debug) System.out.println("relaxing");
		filt_gauss_relax_tensors(tensor, a);
		CONRAD.gc();
		if(debug) System.out.println("normalizing");
		revan[1] = filt_normalize_tensors(tensor, 1);
		CONRAD.gc();

		dimensions = volume.dimensions;
		for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
			size[dim_loop] = volume.size[dim_loop];
			dim[dim_loop] = volume.spacing[dim_loop];
		}

		/* Isotropic LP filter */
		filt=operator.createLowPassFilter(dimensions, size, dim, lpUpper);
		operator.multiply(filt,volume);
		fft.inverseTransform(filt);
		operator.real(filt);      /* OBS !!! filters real and even . filt ~ real */
		revan[0]= filt;

		n_filters = filt_get_n_filters(volume.dimensions);
		Volume3D sum = null;

		for (loop=0; loop<n_filters; loop++) {
			if (showStatus) {
				IJ.showStatus("Assembling High Pass Filters");
				IJ.showProgress((((float)(loop))/n_filters));
			}
			if(debug) System.out.println("high-pass " + loop);
			filt=operator.createHighPassFilter(dimensions, size, dim, loop, lpUpper);
			operator.multiply(filt,volume);
			fft.inverseTransform(filt);
			operator.real(filt);     /* OBS !!! filters real and even . filt ~ real */
			//filt.getImagePlus("test" + loop).show();
			//filt.getImagePlus("filter " + loop).show();
			CONRAD.gc();
			vol=operator.createVolume(size, dim, 1);
			filt_calc_Nk_I(dimensions, loop, t_coeff);
			for (row=0; row<dimensions; row++){
				for (col=row; col<dimensions; col++) {
					vol2=tensor[row][col];
					double weight =  t_coeff[row][col];
					if (row != col) weight *= 2.0;
					operator.addVolume(vol, vol2, weight);
					vol2 = null;
				}
			}
			operator.multiply(vol,filt);
			filt.destroy();
			filt = null;
			if (loop==0){
				sum = vol;
			} else {
				operator.addVolume(sum,vol);
				vol.destroy();
			}
			vol = null;
			CONRAD.gc();
		}
		if (debug) System.out.println("last filter computed ... ");
		/* Clean up */
		for (row=0; row<dimensions; row++){
			for (col=row; col<dimensions; col++) {
				tensor[row][col].destroy();
				tensor[row][col] = null;
			}
		}
		revan[2] = sum;
		return revan;

	}


	public Volume3D [] computeAnisotropicFilteredVolume(Volume3D volume, float low, float high, float hp_lower_level, float hp_upper_level,
			float smth,int A, float B, float ri, float a, float lpUpper)
	{
		Volume3D vol, vol2, orig, sigmoid;
		Volume3D [] pre;
		Volume3D [] revan = new Volume3D[2];

		// save the volume:
		orig = operator.createVolume(volume.size, volume.spacing, volume.in_dim);
		operator.addVolume(orig, volume);
		
		pre = filt_pre_enhance(volume, A, B, ri, a, lpUpper);

		if (debug) System.out.println("after pre_enhance");
		
		// Norm of structure Tensor
		vol2=pre[1];

		
		if (debug) {
			if (vol2 instanceof CUDAVolume3D) ((CUDAVolume3D) vol2).fetch();
		}
		if (debug) System.out.println("before max " + vol2);
		if (debug) System.out.println("before min " + vol2);

		Volume3D copy = operator.createVolume(vol2.size, vol2.spacing, vol2.in_dim);
		operator.addVolume(copy, vol2);
		
		
		/* Sigmoid threshold */
		operator.sigmoid(vol2,smth,low,high,hp_lower_level,hp_upper_level); 

		
		//Volume3D copy = operator.createVolume(vol2.size, vol2.dim, vol2.in_dim);
		//operator.addVolume(copy, vol2);
		
		revan[1]= copy;
		
		if (debug) {
			if (copy instanceof CUDAVolume3D) ((CUDAVolume3D) vol2).fetch();
			vol2.getImagePlus("After Trafo").show();
		}
		
		
		// HP Volume
		vol=pre[2];

		operator.multiply(vol,vol2);
		sigmoid = vol2;
		
		// Low Pass Volume
		vol2=pre[0];
		//vol2.getImagePlus("Low Pass").show();
		operator.addVolume(vol,vol2);
		vol2.destroy();
		
		// Now we take the original values from the orignal image
		// and only the filtered ones from the filtered image.
		operator.multiply(orig, sigmoid);
		operator.multiplyScalar(sigmoid, -1.0f, 0.0f);
		operator.addScalar(sigmoid, 1.0f, 0.0f);
		operator.multiply(vol, sigmoid);
		operator.addVolume(vol, orig);
		
		sigmoid.destroy();
		orig.destroy();
		
		revan[0] = vol;
		
		fft.cleanUp();

		return revan;

	}

	/**
	 * Returns the VolumeOperator of the filter.
	 * @return the VolumeOperator
	 */
	public VolumeOperator getVolumeOperator() {
		return operator;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
