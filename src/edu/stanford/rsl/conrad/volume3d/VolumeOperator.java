package edu.stanford.rsl.conrad.volume3d;

import edu.stanford.rsl.conrad.utils.CONRAD;
import ij.IJ;
import ij.ImagePlus;
import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.*;

/**
 * Class to model operation on volumes. While this class offers many simple operations which can be applied to volumes like add, and multiply, it also contains
 * a suite of operations tailored to filtering volumes based on quadrature filters. 
 * While a quadrature filter is complex in spatial domain, it can often be described using only its real part in frequency domain. Many of the filters
 * used here are based on this feature. The filter functions are modeled as real volumes. However, they are designed to be applied in frequency domain
 * for image processing. All of these filters {@latex.inline %preamble{\\usepackage{amsmath}} $F_\\text{orient}(\\mathbf{u})$} are based on a radial formulation: 
 *   {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
 *   F_\\text{orient}(\\mathbf{u}) & = R(\\rho) \\cdot D_k(\\mathbf{u})
 *   \\end{align*}}
 * {@latex.inline $\\mathbf{u}$} denotes the coordinate in frequency space and {@latex.inline $\\rho = |\\mathbf{u}|$} is its L2-norm.
 * {@latex.inline %preamble{\\usepackage{amsmath}} $D_k(\\mathbf{u})$} controls the isotropy of the filter. For isotropic filters it is simply 1. For anisotropic filters the direction of anisotropy {@latex.inline $\\hat{\\mathbf{n}}_k$} is required:
 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
 * D_k(\\mathbf{u}) & =  \\left \\{ \\begin{array}{ll} \\mathbf{u} \\cdot \\hat{\\mathbf{n}}_k & \\text{if}\\qquad\\mathbf{u} \\cdot \\hat{\\mathbf{n}}_k >0 \\\\ 0 & \\text{else} \\end{array} \\right . \\phantom{\\}}
 * \\end{align*}}
 * The VolumeOperator contains also methods to obtain directionally well spaced {@latex.inline $\\hat{\\mathbf{n}}_k$}.
 * <BR><BR>
 * Note that any method which starts with create will allocate memory. A call of Volume3D.destory() is recommended after it's use.
 * This will free the memory in device-dependent implementations. 
 * 
 * @author akmaier
 *
 */
public class VolumeOperator {

	/**
	 * Sets the behavior of the filter. If it is set to QUADRATIC the filter is computed with directional dependence, i.e. based on the direction vector {@latex.inline $\\hat{\\mathbf{n}}_k$}.
	 * @see VolumeOperator
	 * @author akmaier
	 *
	 */
	public enum FILTER_TYPE { NORMAL, QUADRATIC };

	/**
	 * Turns on the debug mode
	 */
	protected final static boolean DEBUG_FLAG = false;

	/**
	 * is only available in debug mode.
	 * @param s
	 */
	protected static void fprintf(Object s){
		if (DEBUG_FLAG) System.out.println(s);
	}

	/**
	 * Computes the maximal frequency for each dimension as {@latex.inline %preamble{\\usepackage{amsmath}} $f_\\text{max} = \\frac{\\pi}{s_i}$}, where {@latex.inline $s_i$} is the size of each voxel in dimension {@latex.inline $i$} and
	 * the size of each frequency bin  {@latex.inline %preamble{\\usepackage{amsmath}} $f_{\\delta} = \\frac{2 \\cdot f_\\text{max}}{l_i}$}, where {@latex.inline $l_i$} is the number of voxels in dimension {@latex.inline $i$}.<BR>
	 * Method currently applies C-style passing of return values. An array of float values is passed to the method which is filled throughout the call of the method.
	 * @param dimensions the number of dimension of the volume
	 * @param size an array containing {@latex.inline $l_i$}
	 * @param spacing an array containing {@latex.inline $s_i$}
	 * @param f_max the return values for {@latex.inline %preamble{\\usepackage{amsmath}} $f_\\text{max}$}
	 * @param f_delta the return values for {@latex.inline %preamble{\\usepackage{amsmath}} $f_{\\delta}$}
	 */
	public static void getFrequencyBoundings(int dimensions,
			int size[], float spacing[],
			float f_max[], float f_delta[])
	{
		int dim_loop;

		for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
			f_max[dim_loop]   = (float) (Math.PI / spacing[dim_loop]);
			f_delta[dim_loop] = 2*f_max[dim_loop] / size[dim_loop];
		}

		if (DEBUG_FLAG) {

			fprintf("filt_get_boundings\n");

			fprintf("  f max = ");
			for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
				fprintf(f_max[dim_loop]);
				if (dim_loop<dimensions-1)
					fprintf(", ");
			}
			fprintf("\n");
		}

	}

	/**
	 * Method to compute the maximal eigenvalue {@latex.inline %preamble{\\usepackage{amsmath}} $\\lambda_1$} per volume element for a volume structure tensor.
	 * @param structureTensor the structure tensor as 2D array of volumes.
	 * @return the values of {@latex.inline %preamble{\\usepackage{amsmath}} $\\lambda_1$} as volume
	 */
	public Volume3D solveMaximumEigenvalue(Volume3D [][] structureTensor)
	{
		int dimensions = structureTensor[0][0].dimensions;

		float [][] T = new float [Volume3D.MAX_DIM][Volume3D.MAX_DIM];

		Volume3D             vol;
		int                 row, col;
		int [] size = new int [Volume3D.MAX_DIM];
		int in_dim;
		float [] spacing = new float [Volume3D.MAX_DIM];


		if (DEBUG_FLAG)
			fprintf("filt_solve_max_eigenvalue \n");


		/* Get size info from first file */



		size = structureTensor[0][0].size;
		spacing = structureTensor[0][0].spacing;
		in_dim = structureTensor[0][0].getInternalDimension();


		vol= createVolume(size, spacing, in_dim);


		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {


					for (row=0; row<dimensions; row++)
						for (col=row; col<dimensions; col++) {

							T[row][col] = structureTensor[row][col].data[indexX][indexY][indexZ];

							if (row != col) 
								T[col][row] = T[row][col];
						}

					/* solve max eigenvalue ... */
					float [][] v = new float[Volume3D.MAX_DIM][Volume3D.MAX_DIM];
					float [] eig = new float [Volume3D.MAX_DIM];
					int dim_loop;
					Integer nrot = new Integer(0);
					MaxEigenValue.jacobi(T, dimensions, eig, v, nrot);

					float eigmax=eig[0];
					for (dim_loop=1; dim_loop<dimensions; dim_loop++) {
						if (eig[dim_loop]>eigmax) eigmax=eig[dim_loop];
					}

					vol.data[indexX][indexY][indexZ] = eigmax;

				}
			}
		}

		return(vol);
	}

	/**
	 * Creates an anisotropic, i.e. {@latex.inline %preamble{\\usepackage{amsmath}} $D(\\mathbf{u})$} is not constant, filter as Volume3D if FILTER_TYPE is QUADRATIC.
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 * D_k(\\mathbf{u}) & =  \\left \\{ \\begin{array}{ll} \\left(\\frac{(\\displaystyle \\mathbf{u} \\cdot \\hat{\\mathbf{n}}_k)}{\\displaystyle |\\rho|}\\right)^{2A} & \\text{if}\\qquad\\mathbf{u} \\cdot \\hat{\\mathbf{n}}_k >0 \\\\ 0 & \\text{else} \\end{array} \\right . \\phantom{\\}}
	 * \\end{align*}}
	 * {@latex.inline $\\rho = |\\mathbf{u}|$} and {@latex.inline $\\hat{\\mathbf{n}}_k$} is the direction of the anisotropy. The case selection is only valid in the QUADRATIC case.
	 * @param dimensions dimension should be 3
	 * @param size number of pixels per dimension
	 * @param spacing resolution
	 * @param dir direction vector {@latex.inline $\\hat{\\mathbf{n}}_k$}
	 * @param A parameter A
	 * @param t_filt filter type
	 * @return the filter
	 */
	public Volume3D createDirectionalWeights(int dimensions, int size[],
			float spacing[], float dir[], int A, FILTER_TYPE t_filt)
	{
		Volume3D vol;

		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		float r_abs, r_dot, tmp, pos;
		int   exp_loop, dim_loop;

		if (DEBUG_FLAG)
			fprintf("filt_cos2\n");

		vol=createVolume(size, spacing, 1);

		/* normalize filter direction */

		r_abs = 0.0f;
		r_abs += dir[0] * dir[0];
		r_abs += dir[1] * dir[1];
		r_abs += dir[2] * dir[2];
		r_abs = (float) Math.sqrt(r_abs);

		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			dir[dim_loop] /= r_abs;

		if (DEBUG_FLAG) {
			fprintf("  direction = ");
			for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
				fprintf(dir[dim_loop]);
				if (dim_loop<dimensions-1)
					fprintf(", ");
			}
			fprintf("\n");
		}



		/* calculate filter boundings */

		getFrequencyBoundings(dimensions, size, spacing, f_max, f_delta);


		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {

					// r_dot is u * n_k
					r_dot = 0.0f;
					r_abs = 0.0f;

					pos = -f_max[0] + (float) indexX * f_delta[0];
					r_dot += dir[0] * pos;
					r_abs += pos * pos;
					pos = -f_max[1] + (float) indexY * f_delta[1];
					r_dot += dir[1] * pos;
					r_abs += pos * pos;
					pos = -f_max[2] + (float) indexZ * f_delta[2];
					r_dot += dir[2] * pos;
					r_abs += pos * pos;

					// r_abs is the L2 norm of rho
					r_abs = (float) Math.sqrt(r_abs);   /* LW 990320 */

					if (r_abs != 0 && !(t_filt==FILTER_TYPE.QUADRATIC && r_dot < 0) ) {

						tmp = r_dot/r_abs;
						vol.data[indexX][indexY][indexZ] = 1;
						for (exp_loop=0; exp_loop<A; exp_loop++) 
							vol.data[indexX][indexY][indexZ] *= tmp*tmp;

					} else
						vol.data[indexX][indexY][indexZ] = 0;



				}
			}
		}

		//Volume3D.vol_fftshift(vol);

		return(vol);
	}

	/**
	 * Creates a new empty volume according to the parameters. Volume is initialized with zero at all elements.
	 * @param size the sizes in each direction as array
	 * @param spacing the spacing of the pixels in mm in each direction
	 * @param in_dim the internal dimension (1 = real, 2 = complex)
	 * @return the new volume
	 */
	public Volume3D createVolume(int size[],
			float spacing[],
			int in_dim)
	{
		Volume3D volume;

		volume= new Volume3D(size, spacing, in_dim);

		return(volume);
	}

	/**
	 * Creates a Volume3D Object from an ImagePlus. Parameters match the ImagePlus constructor of Volume3D.
	 * If a different type of Volume3D is used (such as CUDAVolume3D) this method is overloaded in the respective inhereted version of of CUDAVolumeOperator.
	 * In order to write CUDA compatible code, use this version instead of the Volume3D constructor.
	 * @param image the ImagePlus with the image data 
	 * @param mirror the number of pixels which are mirrored at the boundary
	 * @param cuty number of pixels to be cut from the ImagePlus (in order to remove emtpy areas at the top and the bottom of the image
	 * @param uneven true if the original number of slices was uneven (may have been altered by cutting y)
	 * @return the volume as Volume3D
	 * @see Volume3D#Volume3D(ImagePlus, int, int, boolean)
	 */
	public Volume3D createVolume(ImagePlus image, int mirror, int cuty, boolean uneven)
	{
		Volume3D volume;

		volume= new Volume3D(image, mirror, cuty, uneven);

		return(volume);
	}

	/**
	 * Creates an radially symetric anisotropic quadrature filter according to this definition:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   F(\\mathbf{u}) & =  \\left \\{ \\begin{array}{ll} \\left(\\frac{\\displaystyle (\\mathbf{u} \\cdot \\hat{\\mathbf{n}}_k)}{\\displaystyle |\\rho|}\\right)^{2A} \\cdot R(\\rho)& \\text{if}\\qquad\\mathbf{u} \\cdot \\hat{\\mathbf{n}}_k >0 \\\\ 0 & \\text{else} \\end{array} \\right . \\phantom{\\}}\\\\
 	 *   R(\\rho) & = e^{\\displaystyle -\\frac{\\displaystyle 4}{\\displaystyle B^2ln(2)}ln^2\\left(\\frac{\\rho}{\\rho_i}\\right)}
	 *   \\end{align*}}<BR>
	 *   The filter is suitable to estimate the local orientation in direction {@latex.inline $\\hat{\\mathbf{n}}_k$}.
	 *   <BR><BR>
	 *   This plot shows {@latex.inline $R(\\rho)$} with B = 2, A = 1, and {@latex.inline $\\rho_i$} = 1.5:<BR><BR>
	 *   <img src="http://peaks.informatik.uni-erlangen.de/KONRAD/exponential_high_pass.gif">
	 *   <BR><BR>Note that the volume is moved to FFT octants with the highest frequencies in the center. It can be directly multiplied to the FFT of a volume.
	 * @see #fftShift(Volume3D) 
	 * @param dimensions dimension should be 3
	 * @param size number of pixels per dimension
	 * @param spacing resolution
	 * @param dir the filter direction {@latex.inline $\\hat{\\mathbf{n}}_k$}
	 * @param A factor A
	 * @param B the relative bandwith of the filter B
	 * @param ri {@latex.inline $\\rho_i$}
	 * @param t_filt parameter to 
	 * @return the filter as volume
	 */
	public Volume3D createExponentialDirectionalHighPassFilter(int dimensions, int size[],
			float spacing[], float dir[], int A, float B, float ri,
			FILTER_TYPE t_filt)
	{
		Volume3D vol;
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		float r_abs, r_dot, tmp, pos;
		int   exp_loop, dim_loop;

		if (DEBUG_FLAG)
			fprintf("filt_cos2_r\n");

		vol=createVolume(size, spacing, 1);

		/* normalize filter direction */

		r_abs=0;
		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			r_abs += dir[dim_loop] * dir[dim_loop];
		r_abs = (float) Math.sqrt(r_abs);

		for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
			dir[dim_loop] /= r_abs;
			if (DEBUG_FLAG)
				fprintf("  direction "+dim_loop+" = "+dir[dim_loop]+"\n");
		}

		/* calculate filter boudings */

		getFrequencyBoundings(dimensions, size, spacing, f_max, f_delta);


		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {

					r_dot = 0;
					r_abs = 0;


					pos = -f_max[0] + indexX * f_delta[0];
					r_dot += dir[0] * pos;
					r_abs += pos * pos;
					pos = -f_max[1] + indexY * f_delta[1];
					r_dot += dir[1] * pos;
					r_abs += pos * pos;
					pos = -f_max[2] + indexZ * f_delta[2];
					r_dot += dir[2] * pos;
					r_abs += pos * pos;

					r_abs = (float) Math.sqrt(r_abs);

					if (r_abs != 0 && !(t_filt==FILTER_TYPE.QUADRATIC && r_dot < 0) ) {

						tmp = r_dot/r_abs;
						vol.data[indexX][indexY][indexZ] = 1;
						for (exp_loop=0; exp_loop<A; exp_loop++) 
							vol.data[indexX][indexY][indexZ] *= tmp*tmp;

						tmp = (float) Math.log(r_abs/ri);
						vol.data[indexX][indexY][indexZ] *= Math.exp( -4.0 / Math.log( (float) 2 ) /  (B*B)*tmp*tmp);

					} else
						vol.data[indexX][indexY][indexZ] = 0;
				}
			}
		}

		fftShift(vol);

		return(vol);
	}

	/**
	 * Creates a directional high pass filter {@latex.inline %preamble{\\usepackage{amsmath}} $F_{\\displaystyle\\text{HP}_k}(\\mathbf{u}) = 1 - (R_\\text{LP}(\\rho) \\cdot D_k(\\mathbf{u}))$}.
	 * The definition of {@latex.inline %preamble{\\usepackage{amsmath}} $D_k(\\mathbf{u})$} from createDirectionalFilter is used. The low-pass filter is defined as:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   R_\\text{LP}(\\rho) & = \\text{cos}^2\\left(\\frac{\\displaystyle \\pi \\cdot \\rho}{\\displaystyle 2 \\rho_\\text{LP}} \\right)
	 *   \\end{align*}}<BR>
	 *   {@latex.inline %preamble{\\usepackage{amsmath}} $\\rho_\\text{LP}$} is the upper limit of the low-pass filter.
	 *   <BR><BR>
	 *   This plot shows the low-pass filter for {@latex.inline %preamble{\\usepackage{amsmath}} $\\rho_\\text{LP}$} = 1.5:<BR>
	 *   <img src="http://peaks.informatik.uni-erlangen.de/KONRAD/cos_low_pass.gif">
	 *   <BR><BR>Note that the volume is moved to FFT octants with the highest frequencies in the center. It can be directly multiplied to the FFT of a volume.
	 * @see #fftShift(Volume3D)  
	 * @param dimensions dimension should be 3
	 * @param size number of pixels per dimension
	 * @param spacing resolution
	 * @param filt_loop {@latex.inline $k$} for the creation of {@latex.inline $\\hat{\\mathbf{n}}_k$}
	 * @param lp_upper {@latex.inline %preamble{\\usepackage{amsmath}} $\\rho_\\text{LP}$}
	 * @return the filter as volume
	 * @see #createDirectionalWeights(int, int[], float[], float[], int, FILTER_TYPE)
	 */
	public Volume3D createHighPassFilter(int dimensions, int [] size, float [] spacing, int filt_loop, float lp_upper){
		float [][] dir = new float[MAX_FILTERS][Volume3D.MAX_DIM];
		float hp_lower = (float) (10f*Math.PI);   /* was PI*10 LW 2006-01-31 */
		float hp_upper = (float) (10f*Math.PI);   /* was PI    LW 2006-01-31 */
		//float lp_upper = 1.50f;    /* was 1.5   LW 2006-01-31 */
		Volume3D vol=createVolume(size, spacing, 1);
		float  r_abs, tmp, pos;
		int    dim_loop, n_filters;

		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		VolumeOperator.getFrequencyBoundings(dimensions, size, spacing, f_max, f_delta);
		/* create HP filters */

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {

					r_abs = 0;

					dim_loop=0;
					pos = -f_max[dim_loop] + (float) indexX * f_delta[dim_loop];
					r_abs += pos * pos;
					dim_loop=1;
					pos = -f_max[dim_loop] + (float) indexY * f_delta[dim_loop];
					r_abs += pos * pos;
					dim_loop=2;
					pos = -f_max[dim_loop] + (float) indexZ * f_delta[dim_loop];
					r_abs += pos * pos;

					r_abs = (float) Math.sqrt(r_abs);

					if (r_abs <= lp_upper) {

						tmp=(float) Math.cos(Math.PI*r_abs/(2.0*lp_upper));
						vol.data[indexX][indexY][indexZ] =	(float) (1.0 - tmp*tmp);

					} else if (lp_upper<r_abs && r_abs<=hp_lower) {

						vol.data[indexX][indexY][indexZ] = 1;

					} else if (hp_lower<r_abs && r_abs<=hp_upper) {

						tmp=(float) Math.cos(Math.PI*(r_abs-hp_lower)/(2.0*(hp_upper-hp_lower)));
						vol.data[indexX][indexY][indexZ] = tmp*tmp;

					} else
						vol.data[indexX][indexY][indexZ] = 0;

				}
			}
		}

		//Volume3D.vol_fftshift(vol);

		filt_get_filt_dirs(vol.dimensions, dir);
		n_filters = filt_get_n_filters(vol.dimensions);

		IJ.showStatus("Computing High Pass Filters");
		IJ.showProgress((((float)(filt_loop))/n_filters));
		Volume3D filt = createDirectionalWeights(vol.dimensions, vol.size, vol.spacing,
				dir[filt_loop] , 1, FILTER_TYPE.NORMAL);

		if (filt==null) {
			fprintf( "filt_make_enhance_filters: Out of memory\n");
			return(null);
		}

		multiply(filt,vol);
		vol.destroy();
		fftShift(filt);
		//filt.getImagePlus("filter"+ filt_loop).show();
		return filt;
	}

	
	/**
	 * Creates an isotropic low-pass filter, i.e. {@latex.inline %preamble{\\usepackage{amsmath}} $D_k(\\mathbf{u})$} = 1.
	 * The radial shape of the filter is determined by:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 * R_\\text{LP}(\\rho) & = \\text{cos}^2\\left(\\frac{\\displaystyle \\pi \\cdot \\rho}{\\displaystyle 2 \\rho_\\text{LP}} \\right)
	 * \\end{align*}}<BR>
	 * This plot shows the low-pass filter for {@latex.inline %preamble{\\usepackage{amsmath}} $\\rho_\\text{LP}$} = 1.5:<BR>
	 * <img src="http://peaks.informatik.uni-erlangen.de/KONRAD/cos_low_pass.gif">
	 * <BR><BR>Note that the volume is moved to FFT octants with the highest frequencies in the center. It can be directly multiplied to the FFT of a volume.
	 * @see #fftShift(Volume3D)  
	 * @param dimensions dimension should be 3
	 * @param size number of pixels per dimension
	 * @param spacing resolution
	 * @param lp_upper {@latex.inline %preamble{\\usepackage{amsmath}} $\\rho_\\text{LP}$}
	 * @return the filter as volume
	 */
	public Volume3D createLowPassFilter(int dimensions, int size[], float spacing [], float lp_upper){
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		Volume3D vol = createVolume(size, spacing, 1);
		float  r_abs, tmp, pos;
		int    dim_loop;
		/* calculate filter boudings */

		VolumeOperator.getFrequencyBoundings(dimensions, size, spacing, f_max, f_delta);

		/* create LP filter */

		//lpUpper = 1.50f;    /* was 1.5   LW 2006-01-31 */

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {

					r_abs = 0;

					dim_loop=0;
					pos = -f_max[dim_loop] + indexX * f_delta[dim_loop];
					r_abs += pos * pos;
					dim_loop=1;
					pos = -f_max[dim_loop] + indexY * f_delta[dim_loop];
					r_abs += pos * pos;
					dim_loop=2;
					pos = -f_max[dim_loop] + indexZ * f_delta[dim_loop];
					r_abs += pos * pos;

					r_abs = (float) Math.sqrt(r_abs);

					if (r_abs <= lp_upper) {
						tmp=(float) Math.cos(Math.PI*r_abs/(2*lp_upper));
						vol.data[indexX][indexY][indexZ] = (tmp*tmp);
					} else
						vol.data[indexX][indexY][indexZ] = 0;


				}
			}
		}


		fftShift(vol);
		return vol;
	}


	/**
	 * Creates an isotropic, i.e. {@latex.inline %preamble{\\usepackage{amsmath}} $D_k(\\mathbf{u})$} = 1, 3-D Gaussian filter as Volume.
	 * The radial shape is determined as:<BR>
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 * R_\\text{Gauss}(\\rho) & = e^{\\displaystyle-\\frac{\\displaystyle\\alpha^2\\rho^2}{\\displaystyle2}}
	 * \\end{align*}}<BR>
	 * <BR>
	 * This plot shows the radial shape for {@latex.inline %preamble{\\usepackage{amsmath}} $\\alpha$} = 0.5, 1.0, 1.5:<BR>
	 * <img src="http://peaks.informatik.uni-erlangen.de/KONRAD/gauss_low_pass.gif">
	 * <BR><BR>Note that the volume is moved to FFT octants with the highest frequencies in the center. It can be directly multiplied to the FFT of a volume.
	 * @see #fftShift(Volume3D)
	 */
	public Volume3D createGaussLowPassFilter(int dimensions, int size[],
			float spacing[], float alpha)
	{
		Volume3D       vol;
		float []       f_max = new float [Volume3D.MAX_DIM];
		float []       f_delta = new float [Volume3D.MAX_DIM];
		double         r_abs_sq, pos;

		if (DEBUG_FLAG)
			fprintf("filt_gauss\n");

		vol=createVolume(size, spacing, 1);

		/* calculate filter boudings */

		VolumeOperator.getFrequencyBoundings(dimensions, size, spacing, f_max, f_delta);


		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {

					r_abs_sq = 0.0;

					pos = - (double) f_max[0] + (double) indexX * (double) f_delta[0];
					r_abs_sq += pos * pos;
					pos = - (double) f_max[1] + (double) indexY * (double) f_delta[1];
					r_abs_sq += pos * pos;
					pos = - (double) f_max[2] + (double) indexZ * (double) f_delta[2];
					r_abs_sq += pos * pos;

					vol.data[indexX][indexY][indexZ] = (float) Math.exp(- (double) 0.5*r_abs_sq*alpha*alpha);

				}
			}	
		}

		fftShift(vol);

		return(vol);
	}

	/**
	 * Determines the arithmetic average {@latex.inline %preamble{\\usepackage{amsmath}} $\\mu$} of all {@latex.inline %preamble{\\usepackage{amsmath}} $N = \\prod_i d_i$} voxels {@latex.inline %preamble{\\usepackage{amsmath}} $x_j$}.
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 * \\mu & = \\frac{1}{N} \\sum_i x_i
	 * \\end{align*}}<BR>
	 * @param vol the volume
	 * @return the mean value {@latex.inline %preamble{\\usepackage{amsmath}} $\\mu$} 
	 */
	public float mean(Volume3D vol)
	{
		float m;

		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("vol_max_pos: Invalid inner dimension\n");
			return(0);
		}

		m = 0.0f;

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.in_dim; indexZ++) {
					m += vol.data[indexX][indexY][indexZ];
				}
			}
		}
		m /= vol.size[0];
		m /= vol.size[1];
		m /= vol.size[2];

		return(m);
	}

	/**
	 * Determines the maximum intensity of a pixel in the given volume.
	 * 
	 * @param vol the volume
	 * @return the maximal value
	 */
	public float max(Volume3D vol)
	{
		float m;

		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("vol_max: Invalid dimension\n");
			return(0);
		}

		m = vol.data[0][0][0];

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.in_dim; indexZ++) {
					if (vol.data[indexX][indexY][indexZ] > m)
						m = vol.data[indexX][indexY][indexZ];
				}
			}
		}

		return(m); 
	}

	/**
	 * Determines the minimum intensity of the volume.
	 * 
	 * @param vol the volume
	 * @return the minimal value
	 */
	public float min(Volume3D vol)
	{
		float m;

		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("vol_max: Invalid dimension\n");
			return(0);
		}

		m = vol.data[0][0][0];

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.in_dim; indexZ++) {
					if (vol.data[indexX][indexY][indexZ] < m)
						m = vol.data[indexX][indexY][indexZ];
				}
			}
		}

		return(m); 
	}

	/**
	 * Multiplies the volume by a scalar. If the volume is real the imaginary part of the scalar should be set to 0.
	 * Any value different from 0 will cause the volume to be converted to complex values.
	 * 
	 * @param vol the volume
	 * @param realPart the real part of the scalar
	 * @param imagPart the imaginary part of the scalar
	 * @return 0, if successful
	 * @see #makeComplex(Volume3D)
	 */
	public int multiplyScalar(Volume3D vol,
			float realPart, 
			float imagPart )
	{
		float tmp_re;

		if (DEBUG_FLAG)
			fprintf("vol_mult_sc\n");

		switch (vol.in_dim) {

		case 1:

			if (imagPart!=0) {
				makeComplex(vol);
				CONRAD.gc();
			}

			if (vol.in_dim==1)

				for (int indexX=0; indexX<vol.size[0]; indexX++) {
					for (int indexY=0; indexY<vol.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
							vol.data[indexX][indexY][indexZ] *= realPart;
						}
					}
				}

			else
				for (int indexX=0; indexX<vol.size[0]; indexX++) {
					for (int indexY=0; indexY<vol.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
							vol.data[indexX][indexY][indexZ*2] *= realPart;
							vol.data[indexX][indexY][indexZ*2+1] *= imagPart;
						}
					}
				}

			break;

		case 2:   

			for (int indexX=0; indexX<vol.size[0]; indexX++) {
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
						tmp_re = vol.data[indexX][indexY][indexZ*2];
						vol.data[indexX][indexY][indexZ*2]   = realPart * vol.data[indexX][indexY][indexZ*2] - imagPart * vol.data[indexX][indexY][indexZ*2+1];
						vol.data[indexX][indexY][indexZ*2+1] = imagPart * tmp_re + realPart * vol.data[indexX][indexY][indexZ*2+1];
					}
				}
			}

			break;

		default:

			fprintf("vol_mult_sc: Invalid dimension\n");
			return(-1);

		}

		return(0);    
	}

	/**
	 * Makes the volume complex, i.e. the internal dimension of the volume is
	 * increased to 2, if required and the data is transfered to interleaved complex
	 * format.
	 * 
	 * @param vol the volume
	 */
	public void makeComplex(Volume3D vol)
	{

		if (vol.in_dim == 2) return;
		if (vol.in_dim != 1) {
			fprintf("vol_make_comlex: Invalid dimension\n");
			return;
		}

		float [][][] temp = new float [vol.size[0]][vol.size[1]][vol.size[2]*2];

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
					temp[indexX][indexY][indexZ*2] = vol.data[indexX][indexY][indexZ];
				}
			}
		}

		vol.data = null;
		vol.data = temp;
		temp = null;
		vol.in_dim = 2;

		return;

	}



	/**
	 * Divides the volume by a scalar. If a real volume is multiplied the imaginary parameter should be set to 0. Otherwise, the volume is made complex.
	 * @param vol the volume
	 * @param realPart the real part of the scalar
	 * @param imagPart the imaginary part of the scalar
	 * @return 0, if successful
	 * @see #makeComplex(Volume3D)
	 */
	public int divideScalar(Volume3D vol,
			float realPart, 
			float imagPart )
	{
		float tmp_re, tmp_im, tmp_abs_sq;

		if (DEBUG_FLAG)
			fprintf("vol_div_sc\n");

		tmp_abs_sq = realPart*realPart+imagPart*imagPart;

		if (tmp_abs_sq == 0) {

			fprintf("vol_div_sc: Division by zero\n");
			return(0);

		}

		tmp_re =  realPart/tmp_abs_sq;
		tmp_im = -imagPart/tmp_abs_sq;

		return(multiplyScalar(vol, tmp_re, tmp_im));
	}

	/**
	 * Adds a scalar to the Volume3D. If the volume is real the imaginary part should be set to 0. Otherwise, the volume is made complex.
	 * @param vol the volume
	 * @param realPart the real part
	 * @param imagPart the imaginary part
	 * @return 0, if successful.
	 * @see #makeComplex(Volume3D)
	 */
	public int addScalar(Volume3D vol,
			float realPart, 
			float imagPart )
	{

		if (DEBUG_FLAG)
			fprintf("vol_add_sc\n");

		switch (vol.in_dim) {

		case 1:

			if (imagPart!=0){
				makeComplex(vol);
				CONRAD.gc();
			}


			if (vol.in_dim == 1)

				for (int indexX=0; indexX<vol.size[0]; indexX++) {
					for (int indexY=0; indexY<vol.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
							vol.data[indexX][indexY][indexZ] += realPart;
						}
					}
				}

			else

				for (int indexX=0; indexX<vol.size[0]; indexX++) {
					for (int indexY=0; indexY<vol.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
							vol.data[indexX][indexY][indexZ*2]   += realPart;
							vol.data[indexX][indexY][indexZ*2+1] += imagPart;
						}
					}
				}

			break;

		case 2:   

			for (int indexX=0; indexX<vol.size[0]; indexX++) {
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
						vol.data[indexX][indexY][indexZ*2]   += realPart;
						vol.data[indexX][indexY][indexZ*2+1] += imagPart;
					}
				}
			}

			break;

		default:

			fprintf("vol_add_sc: Invalid dimension\n");
			return(-1);

		}

		return(0);

	}

	/**
	 * Multiplies two volumes element by element. Volume vol1 is overwritten.
	 * @param vol1 the first volume
	 * @param vol2 the second volume
	 * @return 0, if successful
	 */
	public int multiply(Volume3D vol1, Volume3D vol2)
	{
		float tmp_re1, tmp_re2;
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_mult\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_mult: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim==2 &&  vol2.in_dim==1) {
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){ 
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol1.in_dim>2 || vol2.in_dim>2) {

			fprintf( "vol_mult: Invalid dimension\n");
			return(0);

		}


		switch (vol1.in_dim) {

		case 1:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
						vol1.data[indexX][indexY][indexZ] *=  
							vol2.data[indexX][indexY][indexZ];
					}
				}
			}

			break;

		case 2:   

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
						tmp_re1 = vol1.data[indexX][indexY][indexZ*2];
						tmp_re2 = vol2.data[indexX][indexY][indexZ*2];
						vol1.data[indexX][indexY][indexZ*2]   = 
							vol1.data[indexX][indexY][indexZ*2] * vol2.data[indexX][indexY][indexZ*2]-
							vol1.data[indexX][indexY][indexZ*2+1] * vol2.data[indexX][indexY][indexZ*2+1];
						vol1.data[indexX][indexY][indexZ*2+1] = 
							vol1.data[indexX][indexY][indexZ*2+1]*tmp_re2+
							tmp_re1*vol2.data[indexX][indexY][indexZ*2+1];
					}
				}
			}
			break;

		default:

			fprintf("vol_mult: Invalid dimension\n");
			return(-1);

		}

		return(0);
	}

	/**
	 * Divides the first volume by the second volume element by element
	 * @param vol1 the first volume
	 * @param vol2 the second volume
	 * @return 0, if successful
	 */
	public int divideByVolume(Volume3D vol1, Volume3D vol2)
	{
		float tmp_re1, tmp_re2, tmp_abs_sq;
		int  dim_loop;
		boolean  div_z_mess = false;

		if (DEBUG_FLAG)
			fprintf("vol_div\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_div: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim==2 &&  vol2.in_dim==1) {
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2) {
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol2.in_dim>2) {

			fprintf( "vol_div: Invalid dimension\n");
			return(0);

		}

		switch (vol1.in_dim) {

		case 1:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						if (vol2.data[indexX][indexY][indexZ]==0 && div_z_mess==false) {
							fprintf( "vol_div: Division by zero\n");
							div_z_mess=true;
						} 
						vol1.data[indexX][indexY][indexZ] = vol1.data[indexX][indexY][indexZ] / vol2.data[indexX][indexY][indexZ];

					}
				}
			}

			break;

		case 2:   

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	

						tmp_abs_sq =
							vol2.data[indexX][indexY][indexZ*2]   * vol2.data[indexX][indexY][indexZ*2]+
							vol2.data[indexX][indexY][indexZ*2+1] * vol2.data[indexX][indexY][indexZ*2+1];

						if (tmp_abs_sq==0 && div_z_mess==false) {
							fprintf("vol_div: Division by zero\n");
							div_z_mess=true;
						}

						tmp_re1 = vol1.data[indexX][indexY][indexZ*2];
						tmp_re2 = vol2.data[indexX][indexY][indexZ*2];

						vol1.data[indexX][indexY][indexZ*2]   = 
							(vol1.data[indexX][indexY][indexZ*2]   * vol2.data[indexX][indexY][indexZ*2] +
									vol1.data[indexX][indexY][indexZ*2+1] * vol2.data[indexX][indexY][indexZ*2+1]) / tmp_abs_sq;

						vol1.data[indexX][indexY][indexZ*2+1] = 
							(vol1.data[indexX][indexY][indexZ*2+1] * tmp_re2 -
									vol2.data[indexX][indexY][indexZ*2+1] * tmp_re1) / tmp_abs_sq;

					}
				}
			}

			break;

		default:

			fprintf("vol_div: Invalid dimension\n");
			return(-1);

		}

		return(0);

	}

	/**
	 * Adds the second volume to the first volume.
	 * @param vol1 the first volume
	 * @param vol2 the second volume
	 * @return 0, if successful
	 */
	public int addVolume(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_add\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_add: Volumes have different sizes\n");
				return(-1);

			}

		/* OBS !!! borde inte behova konvertera vol2 . komplex */

		if (vol1.in_dim==2 &&  vol2.in_dim==1) {
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){ 
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol2.in_dim>2 || vol1.in_dim>2) {

			fprintf( "vol_add: Invalid dimension\n");
			return(-1);

		}

		switch (vol1.in_dim) {

		case 1:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						vol1.data[indexX][indexY][indexZ] +=  vol2.data[indexX][indexY][indexZ];

					}
				}
			}

			break;

		case 2:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						vol1.data[indexX][indexY][indexZ*2] +=  vol2.data[indexX][indexY][indexZ*2];
						vol1.data[indexX][indexY][indexZ*2+1] +=  vol2.data[indexX][indexY][indexZ*2+1];
					}
				}
			}

			break;

		default:

			fprintf("vol_div: Invalid dimension\n");
			return(-1);

		}

		return(0);
	}

	/**
	 * Adds the second volume {@latex.inline %preamble{\\usepackage{amsmath}} $y_j$} multiplied by the scalar weight {@latex.inline %preamble{\\usepackage{amsmath}} $s$} to the first volume {@latex.inline %preamble{\\usepackage{amsmath}} $x_j$}.<br>
	 * The first volume is overwritten. The second volume remains unchanged:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 * x_j & := x_j + s \\cdot y_j
	 * \\end{align*}}<BR>
	 * @param vol1 the first volume {@latex.inline %preamble{\\usepackage{amsmath}} $x_j$}
	 * @param vol2 the second volume {@latex.inline %preamble{\\usepackage{amsmath}} $y_j$}
	 * @param weight the weighting factor {@latex.inline %preamble{\\usepackage{amsmath}} $s$}
	 * @return 0, if successful
	 */
	public int addVolume(Volume3D vol1, Volume3D vol2, double weight)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_add\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_add: Volumes have different sizes\n");
				return(-1);

			}

		/* OBS !!! borde inte behova konvertera vol2 . komplex */

		if (vol1.in_dim==2 &&  vol2.in_dim==1){ 
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){ 
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol2.in_dim>2 || vol1.in_dim>2) {

			fprintf( "vol_add: Invalid dimension\n");
			return(-1);

		}

		switch (vol1.in_dim) {

		case 1:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						vol1.data[indexX][indexY][indexZ] +=  vol2.data[indexX][indexY][indexZ] * weight;

					}
				}
			}

			break;

		case 2:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						vol1.data[indexX][indexY][indexZ*2] +=  vol2.data[indexX][indexY][indexZ*2] *weight;
						vol1.data[indexX][indexY][indexZ*2+1] +=  vol2.data[indexX][indexY][indexZ*2+1]*weight;
					}
				}
			}

			break;

		default:

			fprintf("vol_div: Invalid dimension\n");
			return(-1);

		}

		return(0);
	}


	/**
	 * Subtracts the second volume from the first volume. First volume is overwritten.
	 * @param vol1 the first volume
	 * @param vol2 the second volume
	 * @return 0, if successful
	 */
	public int subtractVolume(Volume3D vol1, Volume3D vol2)
	{
		return addVolume(vol1, vol2, -1.0);
	}

	/**
	 * Determines the minimal volume element by element.<br>
	 * The output is stored in the first volume.
	 * 
	 * @param vol1 the first volume
	 * @param vol2 the second volume
	 * @return 0, if successful
	 */
	public int minOfTwoVolumes(Volume3D vol1, Volume3D vol2)
	{
		float tmp1, tmp2;
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_get_min\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_get_min: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim!=1 ||  vol2.in_dim!=1) {

			fprintf( "vol_get_min: Invalid dimension\n");
			return(-1);

		}

		switch (vol1.in_dim) {

		case 1:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						tmp1 = vol1.data[indexX][indexY][indexZ];
						tmp2 = vol2.data[indexX][indexY][indexZ];
						if (tmp1 > tmp2)
							vol1.data[indexX][indexY][indexZ] = tmp2;
					}
				}
			}

			break;

		case 2:

			for (int indexX=0; indexX<vol1.size[0]; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						tmp1 = vol1.data[indexX][indexY][indexZ*2];
						tmp2 = vol2.data[indexX][indexY][indexZ*2];
						if (tmp1 > tmp2)
							vol1.data[indexX][indexY][indexZ*2] = tmp2;
						tmp1 = vol1.data[indexX][indexY][indexZ*2+1];
						tmp2 = vol2.data[indexX][indexY][indexZ*2+1];
						if (tmp1 > tmp2)
							vol1.data[indexX][indexY][indexZ*2+1] = tmp2;
					}
				}
			}

			break;

		default:

			fprintf("vol_div: Invalid dimension\n");
			return(-1);

		}

		return(0);
	}

	/**
	 * Determines the absolute volume of the input volume.<br>
	 * If the input volume is real Math.abs() is called for each element.<br>
	 * If the input volume is complex the power spectrum is computed. <br>
	 * In any case, the resulting volume is real and has internal dimension 1.<br>
	 * Method works in place and overwrites the old input volume.
	 * 
	 * @param vol the input volume.
	 * @return 0, if successful
	 */
	public int abs(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_abs\n");

		if (vol.in_dim != 1 && vol.in_dim != 2) {

			fprintf( "vol_abs: Invalid dimension\n");
			return(-1);
		}



		switch (vol.in_dim) {

		case 1:

			for (int indexX=0; indexX<vol.size[0]; indexX++) {
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {	
						vol.data[indexX][indexY][indexZ] = Math.abs(vol.data[indexX][indexY][indexZ]);
					}
				}
			}
			break;

		case 2:

			float [][][] temp = new float[vol.size[0]][vol.size[1]][vol.size[2]];
			for (int indexX=0; indexX<vol.size[0]; indexX++) {
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {	
						temp[indexX][indexY][indexZ] = (float) Math.sqrt(vol.data[indexX][indexY][indexZ*2]   * vol.data[indexX][indexY][indexZ*2] +
								vol.data[indexX][indexY][indexZ*2+1] * vol.data[indexX][indexY][indexZ*2+1]);
					}
				}
			}

			vol.data = null;
			vol.data = temp;
			temp = null;
			vol.in_dim = 1;

			break;

		}

		return(0);
	}

	/**
	 * Replaces every element in the volume with the output of Math.sqrt(), i.e. each
	 * element's square root.
	 * 
	 * @param vol the volume to be processed.
	 * @return 0, if successful
	 */
	public int sqrt(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_sqrt\n");

		if (vol.in_dim != 1) {

			fprintf( "vol_sqrt: Invalid dimension\n");
			return(-1);
		}

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.in_dim; indexZ++) {	
					vol.data[indexX][indexY][indexZ] = (float) Math.sqrt(vol.data[indexX][indexY][indexZ]);
				}
			}
		}
		return(0);
	}


	/**
	 * Maps volume onto its real part.
	 * @param vol the volume to be mapped.
	 * @return 0, if successful
	 */
	public int real(Volume3D vol)
	{


		if (DEBUG_FLAG)
			fprintf("vol_real\n");

		if (vol.in_dim == 1) return(0);
		if (vol.in_dim != 2) {

			fprintf( "vol_real: Invalid dimension\n");
			return(-1);
		}

		float [][][] temp = new float[vol.size[0]][vol.size[1]][vol.size[2]];
		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {	
					temp[indexX][indexY][indexZ] = vol.data[indexX][indexY][indexZ*2];
				}
			}
		}

		vol.data = null;
		vol.data = temp;
		temp = null;
		vol.in_dim = 1;
		return(0);
	}

	/**
	 * Performs the shift required for the FFT, i.e. moves the high frequencies to the center and the low frequencies to the periphery. If called again on the same volume, its changes are reversed.
	 * @param vol the volume
	 * @return 0, if successful
	 */
	public int fftShift(Volume3D vol)
	{
		float [] tmp_buffer;

		if (vol.in_dim == 1) makeComplex(vol);

		if (DEBUG_FLAG)
			fprintf("vol_fftshift\n");

		tmp_buffer = new float[vol.in_dim];

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]/2; indexZ++) {
					int newX = (indexX + vol.size[0]/2) % vol.size[0];
					int newY = (indexY + vol.size[1]/2) % vol.size[1];
					int newZ = (indexZ + vol.size[2]/2) % vol.size[2];
					for(int i=0;i<vol.in_dim;i++){
						tmp_buffer[i]=vol.data[indexX][indexY][indexZ*vol.in_dim+i];
					}
					for(int i=0;i<vol.in_dim;i++){
						vol.data[indexX][indexY][indexZ*vol.in_dim+i] = vol.data[newX][newY][newZ*vol.in_dim+i];
					}
					for(int i=0;i<vol.in_dim;i++){
						vol.data[newX][newY][newZ*vol.in_dim+i] = tmp_buffer[i];
					}
				}
			}
		}

		return(0);
	}

	/**
	 * Remaps gray values using a sigmoid function. A smoothing factor for the slope of the sigmoid can be set.
	 * lowValue and highValue determine the input values into the sigmoid. highPaddLowerLevel and highPassUpperLevel determine the
	 * output levels of the sigmoid.<BR>
	 * The following figure describes the remapping process:<BR>
	 * <img src="http://peaks.informatik.uni-erlangen.de/KONRAD/sigmoid.gif"><BR>
	 * One configuration in the plot remaps the values 5 to 10 to the range from 0 to 1. If the levels are adjusted, the 
	 * same range can also be remapped to the range from 0.2 to 1.5.
	 * 
	 * @param vol the volume
	 * @param smoothing slope of sigmoid
	 * @param lowValue the lower input limit
	 * @param highValue the upper input limit
	 * @param highPassLowerLevel the lower output value
	 * @param highPassUpperLevel the upper output value
	 * @return 0, if successful
	 */
	public int sigmoid(Volume3D vol,
			float smoothing, float lowValue, float highValue,
			float highPassLowerLevel,  float highPassUpperLevel)
	{
		float f, cc, cl;

		if (DEBUG_FLAG)
			fprintf("vol_sigmoid_th\n");

		if (vol.in_dim != 1) {

			fprintf( "vol_abs: Invalid dimension\n");
			return(-1);
		}


		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.in_dim; indexZ++) {


					if (vol.data[indexX][indexY][indexZ]>highValue)
						vol.data[indexX][indexY][indexZ]=highValue;

					if (vol.data[indexX][indexY][indexZ]<lowValue)
						vol.data[indexX][indexY][indexZ]=lowValue;

					cc = 0.5f*(lowValue+highValue); // center of the input range
					cl = 0.5f*(highPassLowerLevel+highPassUpperLevel); // center of the output range
					f  = 2.0f*Math.abs(vol.data[indexX][indexY][indexZ]-cc); // shift to input range center; abs; 2*
					f  = 1.0f-(f/(highValue-lowValue)); // scale with range
					f  = (float) (1.0f-Math.pow(f,smoothing));
					f  = Math.signum(vol.data[indexX][indexY][indexZ]-cc)*f;
					f  = cl+(f*(highPassUpperLevel-highPassLowerLevel)/2.0f);

					if (f<0.0f) f=0;

					vol.data[indexX][indexY][indexZ]=f;
				}
			}
		}

		return(0);
	}

	/**
	 * Iterates the volume and replaces all entries greater than max with max.
	 * 
	 * @param vol the volume
	 * @param max the maximum
	 * @return 0, if successful
	 */
	public int upperLimit(Volume3D vol, float max)
	{

		if (DEBUG_FLAG)
			fprintf("vol_cut_upper\n");

		if (vol.in_dim != 1) {

			fprintf( "vol_abs: Invalid dimension\n");
			return(-1);
		}

		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.getInternalDimension(); indexZ++) {	
					if (vol.data[indexX][indexY][indexZ] > max)
						vol.data[indexX][indexY][indexZ] = max;
				}
			}
		}

		return(0);
	}

	/**
	 * Maps a complex volume onto its imaginary part.
	 * @param vol the volume
	 * @return 0, if successful
	 */
	public int imag(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_imag\n");

		if (vol.in_dim == 1) return(0);
		if (vol.in_dim != 2) {

			fprintf( "vol_imag: Invalid dimension\n");
			return(-1);
		}

		float [][][] temp = new float[vol.size[0]][vol.size[1]][vol.size[2]];
		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {	
					temp[indexX][indexY][indexZ] = vol.data[indexX][indexY][indexZ*2+1];
				}
			}
		}

		vol.data = null;
		vol.data   = temp;
		temp = null;
		vol.in_dim = 1;

		return(0);
	}
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/