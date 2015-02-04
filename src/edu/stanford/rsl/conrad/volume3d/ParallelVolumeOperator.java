package edu.stanford.rsl.conrad.volume3d;

import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.MAX_FILTERS;
import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.filt_get_filt_dirs;
import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.filt_get_n_filters;
import ij.IJ;
import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.volume3d.operations.AddSlabScalar;
import edu.stanford.rsl.conrad.volume3d.operations.AddSlabs;
import edu.stanford.rsl.conrad.volume3d.operations.CopySlabData;
import edu.stanford.rsl.conrad.volume3d.operations.DivideSlabs;
import edu.stanford.rsl.conrad.volume3d.operations.FFTShifter;
import edu.stanford.rsl.conrad.volume3d.operations.InitializeGaussian;
import edu.stanford.rsl.conrad.volume3d.operations.InitializeHighPass;
import edu.stanford.rsl.conrad.volume3d.operations.InitializeLowPass;
import edu.stanford.rsl.conrad.volume3d.operations.InitializeSquaredCosine;
import edu.stanford.rsl.conrad.volume3d.operations.InitializeSquaredCosineR;
import edu.stanford.rsl.conrad.volume3d.operations.MaxOfSlab;
import edu.stanford.rsl.conrad.volume3d.operations.MeanOfSlab;
import edu.stanford.rsl.conrad.volume3d.operations.MinOfSlab;
import edu.stanford.rsl.conrad.volume3d.operations.MinOfSlabs;
import edu.stanford.rsl.conrad.volume3d.operations.MultiplySlabScalar;
import edu.stanford.rsl.conrad.volume3d.operations.MultiplySlabs;
import edu.stanford.rsl.conrad.volume3d.operations.ParallelVolumeOperation;
import edu.stanford.rsl.conrad.volume3d.operations.SquareRootSlab;
import edu.stanford.rsl.conrad.volume3d.operations.UpperLimitSlab;

public class ParallelVolumeOperator extends VolumeOperator {
	
	@Override
	public Volume3D solveMaximumEigenvalue(Volume3D [][] structureTensor)
	{
		int dimensions = structureTensor[0][0].dimensions;
		MaxEigenValue [] eigenComputer = new MaxEigenValue[CONRAD.getNumberOfThreads()];
		for (int i = 0; i< eigenComputer.length; i++){
			eigenComputer[i] = new MaxEigenValue(dimensions);
			Thread thread = new Thread(eigenComputer[i]);
			thread.start();
		}

		float [][][][] T = new float [eigenComputer.length][structureTensor[0][0].size[2]][Volume3D.MAX_DIM][Volume3D.MAX_DIM];

		Volume3D             vol;
		int                 row, col;
		int [] size = new int [Volume3D.MAX_DIM];
		int in_dim;
		float [] dim = new float [Volume3D.MAX_DIM];


		if (DEBUG_FLAG)
			fprintf("filt_solve_max_eigenvalue");


		/* Get size info from first file */



		size = structureTensor[0][0].size;
		dim = structureTensor[0][0].spacing;
		in_dim = structureTensor[0][0].getInternalDimension();


		vol= createVolume(size, dim, in_dim);
		int currentEigenComputer = 0;
		for (int indexX=0; indexX<vol.size[0]; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
					for (row=0; row<dimensions; row++) {
						for (col=row; col<dimensions; col++) {
							T[currentEigenComputer][indexZ][row][col] = structureTensor[row][col].data[indexX][indexY][indexZ];
							if (row != col) 
								T[currentEigenComputer][indexZ][col][row] = T[currentEigenComputer][indexZ][row][col];
						}
					}
				}
				/* solve max eigenvalue ... */
				eigenComputer[currentEigenComputer].setData(indexX, indexY, size[2], T[currentEigenComputer]);
				currentEigenComputer = (currentEigenComputer +1) % eigenComputer.length;
				while (true){
					if (eigenComputer[currentEigenComputer].done()) break;
					try {
						Thread.sleep(CONRAD.INVERSE_SPEEDUP);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				eigenComputer[currentEigenComputer].getData(vol.data);
			}

		}
		for (int i = 0; i< eigenComputer.length; i++){
			while (true){
				if (eigenComputer[i].done()) break;
				try {
					Thread.sleep(CONRAD.INVERSE_SPEEDUP);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			eigenComputer[i].getData(vol.data);
			eigenComputer[i].terminate();
		}
		return(vol);
	}
	
	@Override
	public Volume3D createDirectionalWeights(int dimensions, int size[],
			float dim[], float dir[], int A, FILTER_TYPE t_filt)
	{
		Volume3D vol;
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		float r_abs;
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("filt_cos2 "+t_filt);

		vol=createVolume(size, dim, 1);

		/* normalize filter direction */

		r_abs = 0.0f;
		r_abs += dir[0] * dir[0];
		r_abs += dir[1] * dir[1];
		r_abs += dir[2] * dir[2];
		r_abs = (float) Math.sqrt(r_abs);

		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			dir[dim_loop] /= r_abs;

	

		/* calculate filter boundings */

		getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		InitializeSquaredCosine cos = new InitializeSquaredCosine();
		cos.setfDelta(f_delta);
		cos.setfMax(f_max);
		cos.setFilterType(t_filt);
		cos.setExponent(A);
		cos.setDirection(dir);
		
		unaryParallelize(cos, vol);
		
		return(vol);
	}
	
	@Override
	public Volume3D createExponentialDirectionalHighPassFilter(int dimensions, int size[],
			float dim[], float dir[], int A, float B, float ri,
			FILTER_TYPE t_filt)
	{
		Volume3D vol;
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		float r_abs;
		int   dim_loop;

		if (DEBUG_FLAG)
			fprintf("filt_cos2_r"+t_filt);

		vol=createVolume(size, dim, 1);

		/* normalize filter direction */

		r_abs=0;
		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			r_abs += dir[dim_loop] * dir[dim_loop];
		r_abs = (float) Math.sqrt(r_abs);

		for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
			dir[dim_loop] /= r_abs;
			
		}

		/* calculate filter boudings */

		getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		InitializeSquaredCosineR cosR = new InitializeSquaredCosineR();
		cosR.setfDelta(f_delta);
		cosR.setfMax(f_max);
		cosR.setFilterType(t_filt);
		cosR.setExponent(A);
		cosR.setDirection(dir);
		cosR.setB(B);
		cosR.setRi(ri);
		
		unaryParallelize(cosR, vol);

		//vol.getImagePlus(dir[0] + " " +dir[1] + " " +dir[2] +" Cosine Square R").show();
		
		fftShift(vol);
		
		//vol.getImagePlus(dir[0] + " " +dir[1] + " " +dir[2] +"Cosine Square R shifted").show();
				
		return(vol);
	}
	
	@Override
	public Volume3D createHighPassFilter(int dimensions, int [] size, float [] dim, int filt_loop, float lp_upper){
		float [][] dir = new float[MAX_FILTERS][Volume3D.MAX_DIM];
		float hp_lower = (float) (10f*Math.PI);   /* was PI*10 LW 2006-01-31 */
		float hp_upper = (float) (10f*Math.PI);   /* was PI    LW 2006-01-31 */
		//float lp_upper = 1.50f;    /* was 1.5   LW 2006-01-31 */
		Volume3D vol=createVolume(size, dim, 1);
		
		int    n_filters;

		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		VolumeOperator.getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);
		/* create HP filters */
		InitializeHighPass hp = new InitializeHighPass();
		hp.setfDelta(f_delta);
		hp.setfMax(f_max);
		hp.setHpLower(hp_lower);
		hp.setHpUpper(hp_upper);
		hp.setLpUpper(lp_upper);
		unaryParallelize(hp,vol);

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
		fftShift(filt);
		//filt.getImagePlus("filter"+ filt_loop).show();
		return filt;
	}
	
	public Volume3D createLowPassFilter(int dimensions, int size[], float dim [], float lp_upper){
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		Volume3D vol = createVolume(size, dim, 1);
		
		/* calculate filter boudings */

		VolumeOperator.getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		/* create LP filter */

		//float lp_upper = 1.50f;    /* was 1.5   LW 2006-01-31 */
		InitializeLowPass lp = new InitializeLowPass();
		lp.setfDelta(f_delta);
		lp.setfMax(f_max);
		lp.setLpUpper(lp_upper);
		unaryParallelize(lp,vol);

		fftShift(vol);
		return vol;
	}
	
	public Volume3D createGaussLowPassFilter(int dimensions, int size[],
			float dim[], float alpha)
	{
		Volume3D       vol;
		float []       f_max = new float [Volume3D.MAX_DIM];
		float []       f_delta = new float [Volume3D.MAX_DIM];
		
		if (DEBUG_FLAG)
			fprintf("filt_gauss");

		vol=createVolume(size, dim, 1);

		/* calculate filter boudings */

		VolumeOperator.getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);


		InitializeGaussian gauss = new InitializeGaussian();
		gauss.setfDelta(f_delta);
		gauss.setfMax(f_max);
		gauss.setSigma(alpha);
		unaryParallelize(gauss, vol);

		fftShift(vol);

		return(vol);
	}
	
	@Override
	public float mean(Volume3D vol)
	{
		float m;
		/* defined for non-complex volumes only */
		if (vol.in_dim != 1) {
			fprintf("parallel vol_max_pos: Invalid inner dimension\n");
			return(0);
		}
		m = 0.0f;
		ParallelVolumeOperation [] ops = unaryParallelize(new MeanOfSlab(), vol);
		for (int indexX=0; indexX<ops.length; indexX++) {
			m += ((Float)ops[indexX].getResult()).floatValue();
		}
		m /= vol.size[0];
		m /= vol.size[1];
		m /= vol.size[2];
		return(m);
	}
	
	@Override
	public float max(Volume3D vol)
	{
		float m;
		/* defined for non-complex volumes only */
		if (vol.in_dim != 1) {
			fprintf("parallel vol_max: Invalid dimension\n");
			return(0);
		}
		m = vol.data[0][0][0];
		ParallelVolumeOperation [] ops = unaryParallelize(new MaxOfSlab(), vol);
		for (int indexX=0; indexX<ops.length; indexX++) {
			float value = ((Float) ops[indexX].getResult()).floatValue();
			if (value > m)
				m = value;
		}
		return(m); 
	}
	
	@Override
	public float min(Volume3D vol)
	{
		float m;

		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("parallel vol_max: Invalid dimension\n");
			return(0);
		}

		m = vol.data[0][0][0];
		ParallelVolumeOperation [] ops = unaryParallelize(new MinOfSlab(), vol);
		for (int indexX=0; indexX<ops.length; indexX++) {
			float value = ((Float) ops[indexX].getResult()).floatValue();
			if (value < m)
				m = value;
		}
		return(m); 
	}
	
	@Override
	public int multiplyScalar(Volume3D vol,
			float re_sc, 
			float im_sc )
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_mult_sc");
		
		if (im_sc!=0) {
			makeComplex(vol);
			CONRAD.gc();
		}

		scalarParallelize( new MultiplySlabScalar(), vol, re_sc, im_sc);

		return(0);    
	}
	
	@Override
	public void makeComplex(Volume3D vol)
	{

		if (vol.in_dim == 2) return;
		if (vol.in_dim != 1) {
			fprintf("parallel vol_make_comlex: Invalid dimension\n");
			return;
		}

		Volume3D temp = new Volume3D(vol.size, vol.spacing, 2);
		
		binaryScalarParallelize(new CopySlabData(), temp, vol, 1.0f, 1.0f);

		vol.data = null;
		vol.data = temp.data;
		temp = null;
		vol.in_dim = 2;

		return;

	}
	
	@Override
	public int addScalar(Volume3D vol,
			float re_sc, 
			float im_sc )
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_add_sc");

		if (im_sc != 0){
			makeComplex(vol);
			CONRAD.gc();
		}
		scalarParallelize(new AddSlabScalar(), vol, re_sc, im_sc);
		return(0);
	}
	
	@Override
	public int multiply(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("parallel vol_mult");

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

		binaryParallelize(new MultiplySlabs(), vol1, vol2);
		
		return(0);
	}
	
	@Override
	public int divideByVolume(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_div");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "parallel vol_div: Volumes have different sizes\n");
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

		binaryParallelize(new DivideSlabs(), vol1, vol2);

		return(0);

	}
	
	@Override
	public int addVolume(Volume3D vol1, Volume3D vol2){
		return addVolume(vol1, vol2, 1.0f);
	}
	
	@Override
	public int addVolume(Volume3D vol1, Volume3D vol2, double weight)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("parallel vol_add");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {
				fprintf( "parallel vol_add: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim==2 &&  vol2.in_dim==1){ 
			makeComplex(vol2);
			if (DEBUG_FLAG) fprintf("Made complex volume 2");
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){
			if(DEBUG_FLAG) {
				fprintf("Volume 1 Z-Dim" + vol1.data[0][0].length);
			}
			makeComplex(vol1);
			if (DEBUG_FLAG) fprintf("Made complex volume 1");
			if(DEBUG_FLAG) {
				fprintf("Volume 1 Z-Dim" + vol1.data[0][0].length);
			}
			CONRAD.gc();
		}

		if (vol2.in_dim>2 || vol1.in_dim>2) {

			fprintf( "parallel vol_add: Invalid dimension\n");
			return(-1);

		}

		binaryScalarParallelize(new AddSlabs(), vol1, vol2, (float) weight, 0f);

		return(0);
	}
	
	
	@Override
	public int minOfTwoVolumes(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("parallel vol_get_min");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "parallel vol_get_min: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim!=1 ||  vol2.in_dim!=1) {

			fprintf( "parallel vol_get_min: Invalid dimension\n");
			return(-1);

		}

		binaryParallelize(new MinOfSlabs(), vol1, vol2);

		return(0);
	}
	
	@Override
	public int abs(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_abs");

		if (vol.in_dim != 1 && vol.in_dim != 2) {

			fprintf( "parallel vol_abs: Invalid dimension\n");
			return(-1);
		}

		Volume3D temp = new Volume3D(vol.size, vol.spacing, 1);
		binaryScalarParallelize(new CopySlabData(), temp, vol, 0.0f, 0.0f);
		vol.data = null;
		vol.data = temp.data;
		vol.in_dim = 1;
		temp = null;

		return(0);
	}
	
	@Override
	public int real(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_real");

		if (vol.in_dim == 1) return(0);
		if (vol.in_dim != 2) {

			fprintf( "vol_real: Invalid dimension\n");
			return(-1);
		}

		Volume3D temp = new Volume3D(vol.size, vol.spacing, 1);
		binaryScalarParallelize(new CopySlabData(), temp, vol, 1.0f, 0.0f);
		vol.data = null;
		vol.data = temp.data;
		vol.in_dim = 1;
		temp = null;
		
		return (0);
	}
	
	@Override
	public int imag(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_real");

		if (vol.in_dim == 1) return(0);
		if (vol.in_dim != 2) {

			fprintf( "parallel vol_real: Invalid dimension\n");
			return(-1);
		}

		Volume3D temp = new Volume3D(vol.size, vol.spacing, 1);
		binaryScalarParallelize(new CopySlabData(), temp, vol, 0.0f, 1.0f);
		vol.data = null;
		vol.data = temp.data;
		vol.in_dim = 1;
		temp = null;
		
		return (0);
	}
	
	

	
	public int upperLimit(Volume3D vol, float max)
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_cut_upper");

		if (vol.in_dim != 1) {

			fprintf( "parallel vol_abs: Invalid dimension\n");
			return(-1);
		}

		scalarParallelize(new UpperLimitSlab(), vol, max, 0f);
		
		return(0);
	}
	
	@Override
	public int sqrt(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("parallel vol_sqrt");

		if (vol.in_dim != 1) {

			fprintf( "parallel vol_sqrt: Invalid dimension\n");
			return(-1);
		}

		unaryParallelize(new SquareRootSlab(), vol);
		
		return(0);
	}
	
	@Override
	public int fftShift(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_fftshift");
		
		if (vol.in_dim == 1) makeComplex(vol);

		unaryParallelize(new FFTShifter(), vol);
		
		return(0);
	}
	
	private ParallelVolumeOperation [] unaryParallelize(ParallelVolumeOperation op, Volume3D vol){
		int numSegments = CONRAD.getNumberOfThreads();
		int segmentSize = (int) Math.ceil(((double)vol.size[0]) / numSegments);
		ParallelVolumeOperation [] operations = new ParallelVolumeOperation[numSegments];
		for (int indexX=0; indexX<numSegments; indexX++) {
			operations[indexX] = op.clone();
			operations[indexX].setBeginIndexX(indexX*segmentSize);
			operations[indexX].setEndIndexX(Math.min((indexX+1)*segmentSize, vol.size[0]));
			operations[indexX].setVol(vol);
		}
		ParallelThreadExecutor exec = new ParallelThreadExecutor(operations);
		exec.setShowStatus(false);
		try {
			exec.execute();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		exec = null;
		return operations;
	}
	
	private ParallelVolumeOperation [] scalarParallelize(ParallelVolumeOperation op, Volume3D vol, float scalar1, float scalar2){
		int numSegments = CONRAD.getNumberOfThreads();
		int segmentSize = (int) Math.ceil(((double)vol.size[0]) / numSegments);
		ParallelVolumeOperation [] operations = new ParallelVolumeOperation[numSegments];
		for (int indexX=0; indexX<numSegments; indexX++) {
			operations[indexX] = op.clone();
			operations[indexX].setBeginIndexX(indexX*segmentSize);
			operations[indexX].setEndIndexX(Math.min((indexX+1)*segmentSize, vol.size[0]));
			operations[indexX].setVol(vol);
			operations[indexX].setScalar1(scalar1);
			operations[indexX].setScalar2(scalar2);
			//operations[indexX].run();
		}
		ParallelThreadExecutor exec = new ParallelThreadExecutor(operations);
		exec.setShowStatus(false);
		try {
			exec.execute();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		exec = null;
		return operations;
	}
	
	private ParallelVolumeOperation [] binaryScalarParallelize(ParallelVolumeOperation op, Volume3D vol1, Volume3D vol2, float scalar1, float scalar2){
		int numSegments = CONRAD.getNumberOfThreads();
		//if (DEBUG_FLAG) System.out.println("Segments: " + numSegments);
		int segmentSize = (int) Math.ceil(((double)vol1.size[0]) / numSegments);
		//if (DEBUG_FLAG) System.out.println("Segment Size " + segmentSize + " Total Size: " + vol1.size[0]);
		ParallelVolumeOperation [] operations = new ParallelVolumeOperation[numSegments];
		for (int indexX=0; indexX<numSegments; indexX++) {
			operations[indexX] = op.clone();
			operations[indexX].setBeginIndexX(indexX*segmentSize);
			operations[indexX].setEndIndexX(Math.min((indexX+1)*segmentSize, vol1.size[0]));
			operations[indexX].setVol1(vol1);
			operations[indexX].setVol2(vol2);
			operations[indexX].setScalar1(scalar1);
			operations[indexX].setScalar2(scalar2);
		}
		ParallelThreadExecutor exec = new ParallelThreadExecutor(operations);
		exec.setShowStatus(false);
		try {
			exec.execute();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		exec = null;
		return operations;
	}
	
	private ParallelVolumeOperation [] binaryParallelize(ParallelVolumeOperation op, Volume3D vol1, Volume3D vol2){
		int numSegments = CONRAD.getNumberOfThreads();
		int segmentSize = (int) Math.ceil(((double)vol1.size[0]) / numSegments);
		ParallelVolumeOperation [] operations = new ParallelVolumeOperation[numSegments];
		for (int indexX=0; indexX<numSegments; indexX++) {
			operations[indexX] = op.clone();
			operations[indexX].setBeginIndexX(indexX*segmentSize);
			operations[indexX].setEndIndexX(Math.min((indexX+1)*segmentSize, vol1.size[0]));
			operations[indexX].setVol1(vol1);
			operations[indexX].setVol2(vol2);
		}
		ParallelThreadExecutor exec = new ParallelThreadExecutor(operations);
		exec.setShowStatus(false);
		try {
			exec.execute();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		exec = null;
		return operations;
	}
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/