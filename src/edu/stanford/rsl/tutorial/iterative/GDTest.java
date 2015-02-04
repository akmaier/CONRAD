package edu.stanford.rsl.tutorial.iterative;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;

public class GDTest {

	private static int GD_STEPSIZE_MAXITER = 20;
	private static float STEPSIZE_FACTOR = 0.9f;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int iter = 60;
		float stepsize = 0.3f;
		float regul = (float) Math.pow(10, -4);
		int size = 100;
		
		new ImageJ();
		
//		Grid3D in = randGrid(size,size,size);
		Grid3D in = new NumericalSheppLogan3D(size,size,size).getNumericalSheppLoganPhantom();
		in.show("In Volume");
		
		Grid3D out = null;
		try {
			out = iterateGradientDescent(in, iter, stepsize, regul);
		} catch (Exception e) {
			e.printStackTrace();
		}
		out.show("GD Out Volume");
	}
	
	private static Grid3D randGrid(int size, int size2, int size3) {
		Grid3D rand = new Grid3D(size,size,size);
		for(int x=0; x<size; ++x)
			for(int y=0; y<size; ++y)
				for(int z=0; z<size; ++z)
					rand.setAtIndex(x,y,z, (float) (Math.random() * 10));
		return rand;
	}

	private static Grid3D iterateGradientDescent(Grid3D myVolIn, int iter, float stepsize, float regul) throws Exception {

		boolean reduceStepsizeAndUpdate = false;
		Grid3D myVol = new Grid3D(myVolIn);
		
		Grid3D gradX, gradY, gradZ;
		
		for (int i = 0; i < iter; ++i) {
			
			if (stepsize < Math.pow(10, -9))
				break;
			
			// G
			boolean offsetLeft = true;
			gradX = GridOp.sub(myVol, myVol, -1, 0, 0, offsetLeft);
			gradY = GridOp.sub(myVol, myVol, 0, -1, 0, offsetLeft);
			gradZ = GridOp.sub(myVol, myVol, 0, 0, -1, offsetLeft);
			
			int numNegVol = GridOp.numNeg(myVol);
			
			// gradMagnitude = sqrt(sum(G.^2,4) + regularization.^2)
			Grid3D gradMag = GridOp.add(GridOp.square(gradX),
					GridOp.square(gradY), GridOp.square(gradZ), regul * regul);
			GridOp.sqrtInPlace(gradMag);

			double tvNorm = GridOp.l1Norm(gradMag);
		
			// upd = divergence(G ./ gradMagnitude) * stepsize;
			// volN = myVol + upd
			
			// normalized gradients
			GridOp.divInPlace(gradX, gradMag);
			GridOp.divInPlace(gradY, gradMag);
			GridOp.divInPlace(gradZ, gradMag);
			
			offsetLeft = false; // offsetRight
			Grid3D gradXTmp = GridOp.sub(gradX, gradX, 1, 0, 0, offsetLeft);
//			fx(1,:,:)   = Px(1,:,:);
//			fx(end,:,:) = -Px(end-1,:,:);
			//float[][][] b = gradXTmp.getBuffer();
			for(int e=0; e<gradXTmp.getSize()[1]; ++e)
				for(int f=0; f<gradXTmp.getSize()[2]; ++f)
					gradXTmp.setAtIndex(0,e,f, gradX.getAtIndex(0,e,f));
			
			
			for(int e=0; e<gradXTmp.getSize()[1]; ++e)
				for(int f=0; f<gradXTmp.getSize()[2]; ++f)
					gradXTmp.setAtIndex(gradXTmp.getSize()[0]-1,e,f, -gradX.getAtIndex(gradXTmp.getSize()[0]-2,e,f));
			
			Grid3D gradYTmp = GridOp.sub(gradY, gradY, 0, 1, 0, offsetLeft);
//			fy(:,1,:)   = Py(:,1,:);
//			fy(:,end,:) = -Py(:,end-1,:);
			//b = gradYTmp.getBuffer();
			for(int e=0; e<gradYTmp.getSize()[0]; ++e)
				for(int f=0; f<gradYTmp.getSize()[2]; ++f)
					gradYTmp.setAtIndex(e,0,f, gradY.getAtIndex(e,0,f));
			for(int e=0; e<gradYTmp.getSize()[0]; ++e)
				for(int f=0; f<gradYTmp.getSize()[2]; ++f)
					gradYTmp.setAtIndex(e,gradYTmp.getSize()[1]-1,f, -gradY.getAtIndex(e,gradYTmp.getSize()[1]-2,f));
			
			Grid3D gradZTmp = GridOp.sub(gradZ, gradZ, 0, 0, 1, offsetLeft);
//			fz(:,:,1)   = Pz(:,:,1);
//			fz(:,:,end) = -Pz(:,:,end-1);
			//b = gradZTmp.getBuffer();
			for(int e=0; e<gradZTmp.getSize()[0]; ++e)
				for(int f=0; f<gradZTmp.getSize()[1]; ++f)
					gradZTmp.setAtIndex(e,f,0, gradZ.getAtIndex(e,f,0));
			for(int e=0; e<gradZTmp.getSize()[0]; ++e)
				for(int f=0; f<gradZTmp.getSize()[1]; ++f)
					gradZTmp.setAtIndex(e,f,gradZTmp.getSize()[2]-1, -gradZ.getAtIndex(e,f,gradZTmp.getSize()[2]-2));
			
			gradX = gradXTmp;
			gradY = gradYTmp;
			gradZ = gradZTmp;
						
			Grid3D upd = GridOp.add(gradX, gradY, gradZ); // divergence
			
			GridOp.mulInPlace(upd, stepsize);
			Grid3D volN = GridOp.add(myVol, upd);
			
			for(int ii=0; 0 == ii || (reduceStepsizeAndUpdate && ii<GD_STEPSIZE_MAXITER) ; ++ii){
				int numNegVolN = GridOp.numNeg(volN);
				if(numNegVolN > numNegVol){
					GridOp.mulInPlace(upd, STEPSIZE_FACTOR );
					stepsize *= STEPSIZE_FACTOR;
					volN = GridOp.add(myVol, upd);
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
				gradMag = GridOp.add(GridOp.square(gradX),
						GridOp.square(gradY), GridOp.square(gradZ), regul
								* regul);
				GridOp.sqrtInPlace(gradMag);

				double tvNormN = GridOp.l1Norm(gradMag);
				if (tvNormN > tvNorm) {
					GridOp.mulInPlace(upd, STEPSIZE_FACTOR);
					stepsize *= STEPSIZE_FACTOR;
					volN = GridOp.add(myVol, upd);
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
