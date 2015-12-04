package edu.stanford.rsl.tutorial.iterative;


import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;

import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;


/**
 * iTV and eTV reconstruction
 * 
 * @author Mario Amrehn
 * GPU Implementation: Daniel Stromer
 * 
 */
public class Etv extends SartCL {
	
	// -----------------------------------------
	protected boolean debugEtv = false;
	// -----------------------------------------
	private float omega, regul;
	private int gdIter;
	private float stepsize;
	private final float STEPSIZE_FACTOR = 0.9f;
	
	private int GD_STEPSIZE_MAXITER = 20;
	private double DYN_GD_MAX_LAMBDA = 1.2;//Double.MAX_VALUE;
	private int DYN_GD_MAX_ITER = 5;
	
	private long timeStart;
	private long timeEnd;

	//grids for gradient descent iterations
	private OpenCLGrid3D gdVol = null;
	private OpenCLGrid3D gradX;
	private OpenCLGrid3D gradY;
	private OpenCLGrid3D gradZ;
	private OpenCLGrid3D gradMag;
	private OpenCLGrid3D gradMagY;
	private OpenCLGrid3D gradMagZ;
	private OpenCLGrid3D divX;
	private OpenCLGrid3D divY;
	private OpenCLGrid3D divZ;
	private OpenCLGrid3D upd;
	private OpenCLGrid3D volN;
	// grids for linear combinations
	private OpenCLGrid3D sartSino;
	private OpenCLGrid3D gdSino;
	private OpenCLGrid3D diffSino;
	private OpenCLGrid3D varGrid;
	private OpenCLGrid3D sartSino2;
	private OpenCLGrid3D projError;
	
	public Etv(int[] volDims, double[] spacing, double[] origin, OpenCLGrid3D oProj, float beta, float omega, int gdIter, float regul, float initStepsize, double dynGD) throws Exception {
		this(volDims, spacing, origin, oProj, beta, omega, gdIter, regul, initStepsize);
		DYN_GD_MAX_LAMBDA = dynGD;
	}
	
	public Etv(int[] volDims, double[] spacing, double[] origin, OpenCLGrid3D oProj, float beta, float omega, int gdIter, float regul, float initStepsize) throws Exception {
		super(volDims, spacing, origin, oProj, beta);
		this.omega = omega;
		this.gdIter = gdIter;
		this.regul = regul;
		this.stepsize = initStepsize;
	}
	
	public Etv(Grid3D initialVol, Grid3D sino, float beta, float omega, int gdIter, float regul, float initStepsize) throws Exception {
		super(initialVol, sino, beta);
		this.omega = omega;
		this.gdIter = gdIter;
		this.regul = regul;
		this.stepsize = initStepsize;
	}
	
	public final void iterateETV() throws Exception{
		iterateETV(1);
	}
	
	private void configure(){
		// queue for opencl
		gdVol 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		
		//gradient descent variables;
		gradX 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		gradY 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		gradZ 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		gradMag 	= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		gradMagY 	= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		gradMagZ 	= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		divX 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		divY 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		divZ 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		volN 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		upd 		= new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		//linear combination
		sartSino 	= new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		gdSino 		= new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		diffSino 	= new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		varGrid 	= new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		sartSino2 	= new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		projError 	= new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		
		//prepare for gpu calculations
		volCL.getDelegate().prepareForDeviceOperation();
		gdVol.getDelegate().prepareForDeviceOperation();
		gradX.getDelegate().prepareForDeviceOperation();
		gradY.getDelegate().prepareForDeviceOperation();
		gradZ.getDelegate().prepareForDeviceOperation();
		gradMag.getDelegate().prepareForDeviceOperation();
		gradMagY.getDelegate().prepareForDeviceOperation();
		gradMagZ.getDelegate().prepareForDeviceOperation();
		divX.getDelegate().prepareForDeviceOperation();
		divY.getDelegate().prepareForDeviceOperation();
		divZ.getDelegate().prepareForDeviceOperation();
		upd.getDelegate().prepareForDeviceOperation();
		volN.getDelegate().prepareForDeviceOperation();
		
		sartSino.getDelegate().prepareForDeviceOperation();
		gdSino.getDelegate().prepareForDeviceOperation();
		oProj.getDelegate().prepareForDeviceOperation();
		varGrid.getDelegate().prepareForDeviceOperation();
		sartSino2.getDelegate().prepareForDeviceOperation();
		diffSino.getDelegate().prepareForDeviceOperation();
		projError.getDelegate().prepareForDeviceOperation();
	}
	
	public void iterateETV(int iter) throws Exception {
		
		configure();
		for(int i=0; i<iter; ++i){
			if(debugEtv) timeStart = System.currentTimeMillis();
			queueIterative.putCopyBuffer(volCL.getDelegate().getCLBuffer(), gdVol.getDelegate().getCLBuffer()).finish();
			gdVol.getDelegate().notifyDeviceChange();

			//SART iteration
			super.iterate();
			
			iterateGradientDescent(gdVol,gdIter);

			float lambda = (float)getLinCombination(volCL, gdVol);
			
			for(int ii=0; lambda>DYN_GD_MAX_LAMBDA && ii<DYN_GD_MAX_ITER; ++ii){
				iterateGradientDescent(gdVol,gdIter);
				lambda = (float)getLinCombination(volCL, gdVol);
			}
			
			if(Double.isNaN(lambda) || Double.isInfinite(lambda)){
				continue; // if vol is empty use SART only
			} else if(1 < lambda)
				lambda = 1;
			/* volume update */
			// NOTE: (1-L)*vol + L*gdVol

			volCL.getGridOperator().multiplyBy(volCL, (float)(1.0-lambda));
			gdVol.getGridOperator().multiplyBy(gdVol, (lambda));
			volCL.getGridOperator().addBySave(volCL, gdVol);
			
			if(debugEtv) {
				timeEnd = System.currentTimeMillis() - timeStart;
				System.out.format("Time gradient descent: %.5f seconds\n", ((double) timeEnd) / 1000.0);
				System.out.println(volCL.getGridOperator().normL1(volCL));
			}
		}
		
		volumeResult = new Grid3D(volCL);
		volumeResult.setOrigin(-geo.getOriginInPixelsX(),-geo.getOriginInPixelsY(),-geo.getOriginInPixelsZ());
		volumeResult.setSpacing(geo.getVoxelSpacingX(),geo.getVoxelSpacingY(),geo.getVoxelSpacingZ());
		unloadEtv();
		
	}

	private double getLinCombination(final OpenCLGrid3D volCL, final OpenCLGrid3D gdVol) throws Exception {
		//creating forward projections of the volumes that have to be compared

		cbp.fastProjectRayDrivenCL(sartSino,volCL);
		cbp.fastProjectRayDrivenCL(gdSino, gdVol);

		sartSino.getGridOperator().fillInvalidValues(sartSino);
		gdSino.getGridOperator().fillInvalidValues(gdSino);
		
		double definedError = (1-omega)*getProjError(sartSino) + omega*getProjError(gdSino);

		//(oProj - sartSino).*diffSino

		queueIterative.putCopyBuffer(gdSino.getDelegate().getCLBuffer(), diffSino.getDelegate().getCLBuffer())
				.putCopyBuffer(oProj.getDelegate().getCLBuffer(), varGrid.getDelegate().getCLBuffer())
				.finish();
		diffSino.getDelegate().notifyDeviceChange();
		varGrid.getDelegate().notifyDeviceChange();
		
		diffSino.getGridOperator().subtractBy(diffSino, sartSino);	
		varGrid.getGridOperator().subtractBy(varGrid, sartSino);		
		varGrid.getGridOperator().multiplyBy(varGrid, diffSino);	
		
		double a = varGrid.getGridOperator().sum(varGrid);		
	
		queueIterative.putCopyBuffer(sartSino.getDelegate().getCLBuffer(), varGrid.getDelegate().getCLBuffer())
				.putCopyBuffer(sartSino.getDelegate().getCLBuffer(), sartSino2.getDelegate().getCLBuffer()).finish();
		varGrid.getDelegate().notifyDeviceChange();
		sartSino2.getDelegate().notifyDeviceChange();
		
		varGrid.getGridOperator().multiplyBy(varGrid, 2);			
		varGrid.getGridOperator().subtractBy(varGrid, gdSino);			
		varGrid.getGridOperator().multiplyBy(varGrid, gdSino);				
		
		sartSino2.getGridOperator().pow(sartSino2, 2);				
		varGrid.getGridOperator().subtractBy(varGrid, sartSino2);	
		double b = varGrid.getGridOperator().sum(varGrid);				

		queueIterative.putCopyBuffer(sartSino.getDelegate().getCLBuffer(), varGrid.getDelegate().getCLBuffer()).finish();
		varGrid.getDelegate().notifyDeviceChange();
		
		varGrid.getGridOperator().multiplyBy(varGrid, 2);
		varGrid.getGridOperator().subtractBy(varGrid, oProj);
		varGrid.getGridOperator().multiplyBySave(varGrid, oProj);
		varGrid.getGridOperator().subtractBy(varGrid, sartSino2);
		double c = varGrid.getGridOperator().sum(varGrid);
		
		return (Math.sqrt(a*a -b*(definedError + c)) + a) / (-b);
	}

	private double getProjError(OpenCLGrid3D sino) throws Exception {
		queueIterative.putCopyBuffer(oProj.getDelegate().getCLBuffer(), projError.getDelegate().getCLBuffer()).finish();
		projError.getDelegate().notifyDeviceChange();
		projError.getGridOperator().subtractBySave(projError, sino);	
		projError.getGridOperator().pow(projError, 2);					
		return  projError.getGridOperator().sum(projError);
	}

	private void iterateGradientDescent(OpenCLGrid3D gdVol, int iter) throws Exception {
		
		boolean reduceStepsizeAndUpdate = false;
		for (int i = 0; i < iter; ++i) {
			
			if (stepsize < Math.pow(10, -9))
				break;
			
			// G
			boolean offsetleft = true;

			
			queueIterative	.putCopyBuffer(gdVol.getDelegate().getCLBuffer(), gradX.getDelegate().getCLBuffer())	
							.putCopyBuffer(gdVol.getDelegate().getCLBuffer(), gradY.getDelegate().getCLBuffer())
							.putCopyBuffer(gdVol.getDelegate().getCLBuffer(), gradZ.getDelegate().getCLBuffer()).finish();
			gradX.getDelegate().notifyDeviceChange();
			gradY.getDelegate().notifyDeviceChange();
			gradZ.getDelegate().notifyDeviceChange();
			
			gradX.getGridOperator().gradX(gradX, gdVol,-1,offsetleft);
			gradY.getGridOperator().gradY(gradY, gdVol,-1,offsetleft);
			gradZ.getGridOperator().gradZ(gradZ, gdVol,-1,offsetleft);

			//not yet implemented in CL

			int numNegVol = gdVol.getGridOperator().countNegativeElements(gdVol);

			// gradMagnitude = sqrt(sum(G.^2,4) + regularization.^2)

			queueIterative.putCopyBuffer(gradX.getDelegate().getCLBuffer(), gradMag.getDelegate().getCLBuffer())	
					.putCopyBuffer(gradY.getDelegate().getCLBuffer(), gradMagY.getDelegate().getCLBuffer())
					.putCopyBuffer(gradZ.getDelegate().getCLBuffer(), gradMagZ.getDelegate().getCLBuffer()).finish();
			gradMag.getDelegate().notifyDeviceChange();
			gradMagY.getDelegate().notifyDeviceChange();
			gradMagZ.getDelegate().notifyDeviceChange();

			gradMag.getGridOperator().pow(gradMag, 2);
			gradMagY.getGridOperator().pow(gradMagY, 2);
			gradMagZ.getGridOperator().pow(gradMagZ, 2);

			gradMag.getGridOperator().addBy(gradMag, gradMagY);
			gradMag.getGridOperator().addBy(gradMag, gradMagZ);
			gradMag.getGridOperator().addBy(gradMag, regul * regul);
			gradMag.getGridOperator().pow(gradMag, 0.5);

			double tvNorm = gradMag.getGridOperator().normL1(gradMag);	// double tvNorm = GridOp.l1Norm(gradMag);

			// upd = divergence(G ./ gradMagnitude) * stepsize;
			// volN = myVol + upd
			
			// normalized gradients
			gradX.getGridOperator().divideBySave(gradX, gradMag);
			gradY.getGridOperator().divideBySave(gradY, gradMag);
			gradZ.getGridOperator().divideBySave(gradZ, gradMag);

			offsetleft = false; // offsetRight
			int offsetvalue = 1;

			// gradMagnitude = sqrt(sum(G.^2,4) + regularization.^2)

			queueIterative.putCopyBuffer(gradX.getDelegate().getCLBuffer(), divX.getDelegate().getCLBuffer())	
					.putCopyBuffer(gradY.getDelegate().getCLBuffer(), divY.getDelegate().getCLBuffer())
					.putCopyBuffer(gradZ.getDelegate().getCLBuffer(), divZ.getDelegate().getCLBuffer()).finish();
			divX.getDelegate().notifyDeviceChange();
			divY.getDelegate().notifyDeviceChange();
			divZ.getDelegate().notifyDeviceChange();

			gradX.getGridOperator().divergence(gradX,divX,offsetvalue,0,0,offsetleft);
			gradY.getGridOperator().divergence(gradY,divY,0,offsetvalue,0,offsetleft);
			gradZ.getGridOperator().divergence(gradZ,divZ,0,0,offsetvalue,offsetleft);
			
			queueIterative.putCopyBuffer(gradX.getDelegate().getCLBuffer(), upd.getDelegate().getCLBuffer()).finish();
			gradX.getDelegate().notifyDeviceChange();		
			upd.getGridOperator().addBySave(upd, gradY);				
			upd.getGridOperator().addBySave(upd, gradZ); 
			upd.getGridOperator().multiplyBy(upd, stepsize);

			queueIterative.putCopyBuffer(gdVol.getDelegate().getCLBuffer(), volN.getDelegate().getCLBuffer()).finish();
			volN.getDelegate().notifyDeviceChange();		
			volN.getGridOperator().addBy(volN, upd);

			for(int ii=0; 0 == ii || (reduceStepsizeAndUpdate && ii<GD_STEPSIZE_MAXITER) ; ++ii){
				int numNegVolN = volN.getGridOperator().countNegativeElements(volN);
				
				if(numNegVolN > numNegVol){
					upd.getGridOperator().multiplyBy(upd, STEPSIZE_FACTOR);
					stepsize *= STEPSIZE_FACTOR;

					queueIterative.putCopyBuffer(gdVol.getDelegate().getCLBuffer(), volN.getDelegate().getCLBuffer()).finish();
					volN.getDelegate().notifyDeviceChange();		
					
					volN.getGridOperator().addBy(volN, upd);
					reduceStepsizeAndUpdate = true;
					continue;
				}
				// check stepsize
				// GN
				offsetleft = true;

				queueIterative.putCopyBuffer(volN.getDelegate().getCLBuffer(), gradX.getDelegate().getCLBuffer())	
						.putCopyBuffer(volN.getDelegate().getCLBuffer(), gradY.getDelegate().getCLBuffer())
						.putCopyBuffer(volN.getDelegate().getCLBuffer(), gradZ.getDelegate().getCLBuffer()).finish();
				gradX.getDelegate().notifyDeviceChange();
				gradY.getDelegate().notifyDeviceChange();
				gradZ.getDelegate().notifyDeviceChange();
				
				gradX.getGridOperator().gradX(gradX, volN,-1,offsetleft);
				gradY.getGridOperator().gradY(gradY, volN,-1,offsetleft);
				gradZ.getGridOperator().gradZ(gradZ, volN,-1,offsetleft);
				// gradMagnitudeN = sqrt(sum(GN.^2,4) + regularization.^2)

				queueIterative.putCopyBuffer(gradX.getDelegate().getCLBuffer(), gradMag.getDelegate().getCLBuffer())	
						.putCopyBuffer(gradY.getDelegate().getCLBuffer(), gradMagY.getDelegate().getCLBuffer())
						.putCopyBuffer(gradZ.getDelegate().getCLBuffer(), gradMagZ.getDelegate().getCLBuffer()).finish();
				gradMag.getDelegate().notifyDeviceChange();
				gradMagY.getDelegate().notifyDeviceChange();
				gradMagZ.getDelegate().notifyDeviceChange();

				gradMag.getGridOperator().pow(gradMag, 2);
				gradMagY.getGridOperator().pow(gradMagY, 2);
				gradMagZ.getGridOperator().pow(gradMagZ, 2);
				
				gradMag.getGridOperator().addBySave(gradMag, gradMagY);
				gradMag.getGridOperator().addBySave(gradMag, gradMagZ);
		
				gradMag.getGridOperator().addBy(gradMag, regul * regul);
				gradMag.getGridOperator().pow(gradMag, 0.5);
	
				double tvNormN = gradMag.getGridOperator().normL1(gradMag);

				if (tvNormN > tvNorm) {
					upd.getGridOperator().multiplyBySave(upd, STEPSIZE_FACTOR);
					stepsize *= STEPSIZE_FACTOR;
					
					queueIterative.putCopyBuffer(gdVol.getDelegate().getCLBuffer(), volN.getDelegate().getCLBuffer()).finish();
					volN.getDelegate().notifyDeviceChange();
					
					volN.getGridOperator().addBySave(volN, upd);
					tvNorm = tvNormN;
					reduceStepsizeAndUpdate = true;
					continue;
				}
				reduceStepsizeAndUpdate = false;

			}

			queueIterative.putCopyBuffer(volN.getDelegate().getCLBuffer(), gdVol.getDelegate().getCLBuffer()).finish();
			gdVol.getDelegate().notifyDeviceChange();
		}
	}
	
	private void unloadEtv(){
		if(queueIterative != null && !queueIterative.isReleased()) queueIterative.release();
			//grids for gradient descent iterations
		if(gdVol != null ) gdVol.release();
		if(gradX != null ) gradX.release();
		if(gradY != null ) gradY.release();
		if(gradZ != null ) gradZ.release();
		if(gradMag != null) gradMag.release();
		if(gradMagY != null) gradMagY.release();
		if(gradMagZ != null) gradMagZ.release();
		if(divX != null) divX.release();
		if(divY != null) divY.release();
		if(divZ != null) divZ.release();
		if(upd != null) upd.release();
		if(volN != null) volN.release();
		// grids for linear combinations
		if(sartSino != null) sartSino.release();
		if(gdSino != null) gdSino.release();
		if(diffSino != null) diffSino.release();
		if(varGrid != null) varGrid.release();
		if(sartSino2 != null) sartSino2.release();
		if(projError != null) projError.release();
	}
	
	public static void main(String[] args){
		
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory traj = conf.getGeometry();

		new ImageJ();

		ConeBeamProjector cbp = new ConeBeamProjector();

		OpenCLGrid3D grid = new OpenCLGrid3D(new NumericalSheppLogan3D(traj.getReconDimensionX(),traj.getReconDimensionY(), traj.getReconDimensionZ()).getNumericalSheppLoganPhantom());
		
		grid.setOrigin(-traj.getOriginInPixelsX(),-traj.getOriginInPixelsY(),-traj.getOriginInPixelsZ());
		grid.setSpacing(traj.getVoxelSpacingX(),traj.getVoxelSpacingY(),traj.getVoxelSpacingZ());

		try {
			
			final 	float 	omega 			= 0.3f;
			final 	int 	gdIter 			= 25;
			final 	float 	regul 			= (float) Math.pow(10, -4);
			final  	float 	initStepsize	= 0.3f;
			final  	int 	eTVIterations	= 2;
			final  	float 	beta			= 0.8f;

			OpenCLGrid3D sino = new OpenCLGrid3D(new Grid3D(traj.getDetectorWidth(),traj.getDetectorHeight(),traj.getProjectionStackSize()));
			sino.setOrigin(0,0,0);
			sino.setSpacing(1,1,1);
			cbp.fastProjectRayDrivenCL(sino,grid);
			
			System.out.println("GT: "+sino.getGridOperator().normL1(grid));
			Etv etvtest = new Etv(grid.getSize(),grid.getSpacing(),grid.getOrigin(), sino, beta, omega, gdIter, regul,initStepsize);

			long start = System.currentTimeMillis();
			etvtest.iterateETV(eTVIterations);
			long ende = System.currentTimeMillis()-start;
			System.out.format("Time iTV: %.5f seconds\n", ((double) ende) / 1000.0);
			Grid3D foo = etvtest.getVol();
			System.out.println("L1: "+foo.getGridOperator().normL1(foo));
			System.out.println("RMSE: "+foo.getGridOperator().rmse(foo,grid));
			foo.show();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
