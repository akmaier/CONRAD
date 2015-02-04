package edu.stanford.rsl.tutorial.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;

/**
 * iTV and eTV reconstruction
 * 
 * @author Mario Amrehn
 * 
 */
public class Etv extends SartCPU {
	
	// -----------------------------------------
	protected boolean verbose = false;
	// -----------------------------------------
	
	private float omega, regul;
	private int gdIter;
	private float stepsize;
	private final float STEPSIZE_FACTOR = 0.9f;
	
	private Grid3D gradX = null, gradY = null, gradZ = null;
	private int GD_STEPSIZE_MAXITER = 20;
	private double DYN_GD_MAX_LAMBDA = Double.MAX_VALUE;
	private int DYN_GD_MAX_ITER = 5;
	
	public Etv(int[] volDims, double[] spacing, double[] origin, Grid3D oProj, float beta, float omega, int gdIter, float regul, float initStepsize, double dynGD) throws Exception {
		this(volDims, spacing, origin, oProj, beta, omega, gdIter, regul, initStepsize);
		DYN_GD_MAX_LAMBDA = dynGD;
	}
	
	public Etv(int[] volDims, double[] spacing, double[] origin, Grid3D oProj, float beta, float omega, int gdIter, float regul, float initStepsize) throws Exception {
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
	
	public void iterateETV(int iter) throws Exception {
		Grid3D gdVol = null;
		for(int i=0; i<iter; ++i){
			super.iterate(); // SART // TODO
			gdVol = iterateGradientDescent(vol, gdIter);
			double lambda = getLinCombination(vol, gdVol);
			for(int ii=0; lambda>DYN_GD_MAX_LAMBDA && ii<DYN_GD_MAX_ITER; ++ii){
				gdVol = iterateGradientDescent(vol, gdIter);
				lambda = getLinCombination(vol, gdVol);
			}
			if(Double.isNaN(lambda) || Double.isInfinite(lambda)){
				if(verbose) System.out.println("Lambda was invalid");
				continue; // if vol is empty use SART only
			} else if(1 < lambda)
				lambda = 1;
			
			/* volume update */
			// NOTE: (1-L)*vol + L*gdVol
			gop.multiplyBy(vol, (float)(1.0-lambda));
			gop.multiplyBy(gdVol, (float)(lambda));
			gop.addBySave(vol, gdVol);
		}
	}

	private double getLinCombination(Grid3D sartVol, Grid3D gdVol) throws Exception {
		ConeBeamProjector cbp = new ConeBeamProjector();
		//Grid3D sartSino = USE_CL_FP ? cbp.projectRayDrivenCL(sartVol) : cbp.projectPixelDriven(sartVol);
		Grid3D sartSino = cbp.projectRayDrivenCL(sartVol);
		//Grid3D gdSino = USE_CL_FP ? cbp.projectRayDrivenCL(gdVol) : cbp.projectPixelDriven(gdVol);
		Grid3D gdSino = cbp.projectRayDrivenCL(gdVol);
		
		if (verbose) reportInvalidValues(sartSino, "sartSino");
		if (verbose) reportInvalidValues(gdSino, "gdSino");
		
		gop.fillInvalidValues(sartSino);
		gop.fillInvalidValues(gdSino);
		
		double definedError = (1-omega)*getProjError(sartSino) + omega*getProjError(gdSino);
		
		Grid3D diffSino = new Grid3D(gdSino);	//
		gop.subtractBy(diffSino, sartSino);		//Grid3D diffSino = GridOp.sub(gdSino, sartSino);
		
		if (verbose) reportInvalidValues(diffSino, "diffSino");
		
		//(oProj - sartSino).*diffSino
		Grid3D gridA = new Grid3D(oProj);		//
		gop.subtractBy(gridA, sartSino);		//
		gop.multiplyBy(gridA, diffSino);		//
		if (verbose) reportInvalidValues(gridA, "gridA");
		double a = gop.sum(gridA);			// double a = GridOp.sum(GridOp.mulInPlace(GridOp.sub(oProj, sartSino), diffSino));
		
		//sum(gdSino .* (2*sartSino - gdSino) - sartSinoSquared)
		Grid3D gridB = new Grid3D(sartSino);	//
		gop.multiplyBy(gridB, 2);				//
		gop.subtractBy(gridB, gdSino);			//
		gop.multiplyBy(gridB, gdSino);			//
		
		if (verbose) reportInvalidValues(gridB, "gridB before");
		
		gdSino = null;
		Grid3D sartSinoSquared = new Grid3D(sartSino);	//
		gop.pow(sartSinoSquared, 2);					// Grid3D sartSinoSquared = GridOp.square(sartSino);
		
		gop.subtractBy(gridB, sartSinoSquared);	//
		
		if (verbose) reportInvalidValues(gridB, "gridB after");
		
		double b = gop.sum(gridB);				//double b = GridOp.sum(GridOp.sub(GridOp.mul(gdSino, GridOp.sub(GridOp.mul(sartSino, 2), gdSino)), sartSinoSquared));
		gridB = null;
		
		Grid3D gridC = sartSino; //new Grid3D(sartSino);
		gop.multiplyBy(gridC, 2);
		gop.subtractBy(gridC, oProj);
		gop.multiplyBySave(gridC, oProj);
		gop.subtractBy(gridC, sartSinoSquared);
		
		if (verbose) reportInvalidValues(gridC, "gridC");
		
		double c = gop.sum(gridC);
		gridC = null;
		/*
		double c = 
				GridOp.sum(
						GridOp.sub(
								GridOp.mul(oProj, GridOp.sub(GridOp.mul(sartSino, 2), oProj)), 
								sartSinoSquared
						)
				);
		*/
		double lambda = (Math.sqrt(a*a -b*(definedError + c)) + a) / (-b);
		if(verbose && Double.isInfinite(lambda) && Double.isNaN(lambda))
			System.err.println("Error: Lambda is invalid!");
		return lambda;
	}

	private double getProjError(Grid3D sino) throws Exception {
		Grid3D tmp = new Grid3D(oProj);		//
		gop.subtractBySave(tmp, sino);	//
		gop.pow(tmp, 2);					//
		double e = gop.sum(tmp);			// GridOp.sum(GridOp.square(GridOp.sub(oProj, sino)));
		return e;
	}

	private Grid3D iterateGradientDescent(Grid3D myVolIn, int iter) throws Exception {

		boolean reduceStepsizeAndUpdate = false;
		Grid3D myVol = new Grid3D(myVolIn, true);
		
		for (int i = 0; i < iter; ++i) {
			
			if (stepsize < Math.pow(10, -9))
				break;
			
			// G
			boolean offsetLeft = true;
			gradX = GridOp.sub(myVol, myVol, -1, 0, 0, offsetLeft);
			gradY = GridOp.sub(myVol, myVol, 0, -1, 0, offsetLeft);
			gradZ = GridOp.sub(myVol, myVol, 0, 0, -1, offsetLeft);
			
			int numNegVol = gop.countNegativeElements(myVol);
			
			// gradMagnitude = sqrt(sum(G.^2,4) + regularization.^2)
			Grid3D gradMag = new Grid3D(gradX);
			gop.pow(gradMag, 2);
			Grid3D gradMagY = new Grid3D(gradY);
			gop.pow(gradMagY, 2);
			Grid3D gradMagZ = new Grid3D(gradZ);
			gop.pow(gradMagZ, 2);
			gop.addBy(gradMag, gradMagY);
			gop.addBy(gradMag, gradMagZ);
			gop.addBy(gradMag, regul * regul);
			gop.pow(gradMag, 0.5);
			/*
			Grid3D gradMag = GridOp.add(GridOp.square(gradX),
					GridOp.square(gradY), GridOp.square(gradZ), regul * regul);
			GridOp.sqrtInPlace(gradMag);
			*/
			
			double tvNorm = gop.normL1(gradMag);	// double tvNorm = GridOp.l1Norm(gradMag);
		
			// upd = divergence(G ./ gradMagnitude) * stepsize;
			// volN = myVol + upd
			
			// normalized gradients
			gop.divideBySave(gradX, gradMag);
			gop.divideBySave(gradY, gradMag);
			gop.divideBySave(gradZ, gradMag);
			
			offsetLeft = false; // offsetRight
			Grid3D gradXTmp = GridOp.sub(gradX, gradX, 1, 0, 0, offsetLeft);
//			fx(1,:,:)   = Px(1,:,:);
//			fx(end,:,:) = -Px(end-1,:,:);
			//float[][][] b = gradXTmp.getBuffer();
			for(int e=0; e<gradXTmp.getSize()[1]; ++e)
				for(int f=0; f<gradXTmp.getSize()[2]; ++f){
					gradXTmp.setAtIndex(0,e,f, gradX.getAtIndex(0,e,f));
					gradXTmp.setAtIndex(gradXTmp.getSize()[0]-1,e,f, -gradX.getAtIndex(gradXTmp.getSize()[0]-2,e,f));
				}
			
			Grid3D gradYTmp = GridOp.sub(gradY, gradY, 0, 1, 0, offsetLeft);
//			fy(:,1,:)   = Py(:,1,:);
//			fy(:,end,:) = -Py(:,end-1,:);
			//b = gradYTmp.getBuffer();
			for(int e=0; e<gradYTmp.getSize()[0]; ++e)
				for(int f=0; f<gradYTmp.getSize()[2]; ++f){
					gradYTmp.setAtIndex(e,0,f, gradY.getAtIndex(e,0,f));
					gradYTmp.setAtIndex(e,gradYTmp.getSize()[1]-1,f, -gradY.getAtIndex(e,gradYTmp.getSize()[1]-2,f));
				}
			
			Grid3D gradZTmp = GridOp.sub(gradZ, gradZ, 0, 0, 1, offsetLeft);
//			fz(:,:,1)   = Pz(:,:,1);
//			fz(:,:,end) = -Pz(:,:,end-1);
			//b = gradZTmp.getBuffer();
			for(int e=0; e<gradZTmp.getSize()[0]; ++e)
				for(int f=0; f<gradZTmp.getSize()[1]; ++f){
					gradZTmp.setAtIndex(e,f,0, gradZ.getAtIndex(e,f,0));
					gradZTmp.setAtIndex(e,f,gradZTmp.getSize()[2]-1, -gradZ.getAtIndex(e,f,gradZTmp.getSize()[2]-2));
				}
			
			gradX = gradXTmp;
			gradY = gradYTmp;
			gradZ = gradZTmp;
			
			Grid3D upd = new Grid3D(gradX);			//
			gop.addBySave(upd, gradY);					//
			gop.addBySave(upd, gradZ); // divergence	// Grid3D upd = GridOp.add(gradX, gradY, gradZ); // divergence
			gop.multiplyBy(upd, stepsize);
			
			Grid3D volN = new Grid3D(myVol);
			gop.addBy(volN, upd);
			
			for(int ii=0; 0 == ii || (reduceStepsizeAndUpdate && ii<GD_STEPSIZE_MAXITER) ; ++ii){
				int numNegVolN = gop.countNegativeElements(volN);
				if(numNegVolN > numNegVol){
					gop.multiplyBy(upd, STEPSIZE_FACTOR);
					stepsize *= STEPSIZE_FACTOR;
					volN = new Grid3D(myVol);
					gop.addBy(volN, upd);
					reduceStepsizeAndUpdate = true;
					continue;
				}
				// check stepsize
				// GN
				offsetLeft = true;
				gradX = GridOp.sub(volN, volN, -1, 0, 0, offsetLeft);
				gradY = GridOp.sub(volN, volN, 0, -1, 0, offsetLeft);
				gradZ = GridOp.sub(volN, volN, 0, 0, -1, offsetLeft);

				// gradMagnitudeN = sqrt(sum(GN.^2,4) + regularization.^2)
				gradMag = new Grid3D(gradX);
				gop.pow(gradMag, 2);
				gradMagY = new Grid3D(gradY);
				gop.pow(gradMagY, 2);
				gradMagZ = new Grid3D(gradZ);
				gop.pow(gradMagZ, 2);
				gop.addBySave(gradMag, gradMagY);
				gop.addBySave(gradMag, gradMagZ);
				gop.addBy(gradMag, regul * regul);
				gop.pow(gradMag, 0.5);
				/*
				gradMag = GridOp.add(GridOp.square(gradX),
						GridOp.square(gradY), GridOp.square(gradZ), regul
								* regul);
				GridOp.sqrtInPlace(gradMag);
				*/
				
				double tvNormN = gop.normL1(gradMag);
				if (tvNormN > tvNorm) {
					gop.multiplyBySave(upd, STEPSIZE_FACTOR);
					stepsize *= STEPSIZE_FACTOR;
					volN = new Grid3D(myVol);
					gop.addBySave(volN, upd);
					tvNorm = tvNormN;
					reduceStepsizeAndUpdate = true;
					continue;
				}
				
				reduceStepsizeAndUpdate = false;
			}
			myVol = volN;
		}
		return myVol;
	}
	
}
