package edu.stanford.rsl.science.iterativedesignstudy;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;

public class PerformReconstruction {
	public static Projector projector;
	public static Backprojector backprojector;
	public static ART art;



	public static void main(String[] args) {
		new ImageJ();
		int x = 200;
		int y = 200;
		Phantom phan = new SheppLogan(x, false);
		phan.show("The Phantom");
		
		
		//0 = Parallel
		//1 = Fan Beam
		//2 = Parallel + StepSize Control
		//3 = Fan + StepSize Control
		//4 = Parallel Beam with OpenCL
		//5 = Parallel Beam with OpenCL with StepSize
		//6 = Fan Bean with OpenCL
		//7 = Fan Bean with OpenCL with StepSize
		int type = 4; 
		int numberOfSubsets = 10;
		double epsilon = 0.00001;
		int maxIter = 100;		
		
		Grid2D originalSinogram;
		Grid1D sino;
		Grid2D diff2D;
		Grid2D gradientSinogram;
		Grid2D recon = new Grid2D(x, y);
		recon.setSpacing(1, 1);
		Grid2D imageUpdate = new Grid2D(x,y);
		imageUpdate.setSpacing(1, 1);
		Grid2D localImageUpdate = new Grid2D(x,y);
		localImageUpdate.setSpacing(1, 1);
		
		// Parallel geometry params
		//
		//maxTheta = angular range in radian
		double maxTheta = Math.PI; 		
		//deltaTheta  = angluara step size in radian
		double deltaTheta = Math.PI/180;
		//maxS = detector size in mm
		double maxS = 400;
		//deltaS = detector element size in mm
		double deltaS = 1;
		int maxSIndex = (int) (maxS / deltaS + 1);
		int maxThetaIndex = (int) (maxTheta / deltaTheta + 1);
		
		// Fan beam geometry params
		//
		double gammaM = 11.768288932020647*Math.PI/180; 
		//maxT = length of the detector array
		double maxT = 500; 
		//deltaT = size of one detector element
		double deltaT = 1.0; 
		//focalLength = focal length
		double focalLength = (maxT/2.0-0.5)*deltaT/Math.tan(gammaM);
		//maxBeta = maximum rotation angle
		double maxBeta=2*Math.PI;		
		//deltaBeta = step size between source positions
		double deltaBeta=maxBeta/360;
		int maxTIndex = (int) (maxT / deltaT + 1);
		int maxBetaIndex = (int) (maxBeta / deltaBeta + 1);
//		long time=System.nanoTime();
		long startTime = System.currentTimeMillis();


		
		switch (type) {
		case (0): //Parallel	
//			System.out.println("Parallel Beam, Subsets: " + numberOfSubsets);

			originalSinogram = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
			originalSinogram.setSpacing(1, 1.0);

			sino = new Grid1D(new float[maxSIndex]);
			
			projector = new ParallelProjectorRayDriven(maxTheta, deltaTheta, maxS, deltaS);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
			originalSinogram.show("Sino");
			backprojector = new ParallelBackprojectorPixelDriven(originalSinogram);	
			
			art = new ART(projector, backprojector);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);


			break;

		case (1): // Fan Beam
//			System.out.println("Fan Beam, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(maxTIndex, maxBetaIndex);
			originalSinogram.setSpacing(deltaT, deltaBeta);
	
			sino = new Grid1D(new float[maxTIndex]);
			
			projector = new FanProjectorRayDriven(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
			backprojector = (Backprojector) new FanBackprojectorPixelDriven(focalLength, x, y, originalSinogram);
			
			art = new ART(projector, backprojector);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

	
			break;
			
		case (2): //Parallel + StepSize Control	
//			System.out.println("Parallel Beam, Step size control, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
			originalSinogram.setSpacing(1, 1.0);

			diff2D = new Grid2D(originalSinogram.getWidth(), (int)maxThetaIndex);
			diff2D.setSpacing(1, 1);

			gradientSinogram = new Grid2D(originalSinogram.getWidth(), originalSinogram.getHeight());
			gradientSinogram.setSpacing(1, 1);
			
			sino = new Grid1D(new float[maxSIndex]);
			
			projector = new ParallelProjectorRayDriven(maxTheta, deltaTheta, maxS, deltaS);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
			originalSinogram.show("Sino");
			backprojector = new ParallelBackprojectorPixelDriven(originalSinogram);			
						
			art = new ARTStepSizeControl(projector, backprojector, diff2D, gradientSinogram);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

			break;
			
		case (3): //Fan + StepSize Control
//			System.out.println("Fan Beam, Step size control, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(maxTIndex, maxBetaIndex);
			originalSinogram.setSpacing(deltaT, deltaBeta);

			diff2D = new Grid2D(originalSinogram.getWidth(), (int)maxBetaIndex);
			diff2D.setSpacing(deltaT, deltaBeta);

			gradientSinogram = new Grid2D(originalSinogram.getWidth(), originalSinogram.getHeight());
			gradientSinogram.setSpacing(deltaT, deltaBeta);
			
			sino = new Grid1D(new float[maxTIndex]);
			
			projector = new FanProjectorRayDriven(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
//			originalSinogram.show("Sino");
			backprojector = (Backprojector) new FanBackprojectorPixelDriven(focalLength, x, y, gradientSinogram);			
			
			art = new ARTStepSizeControl(projector, backprojector, diff2D, gradientSinogram);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

			break;
			
		case (4)://Parallel Beam with OpenCL
//			System.out.println("Parallel Beam OpenCL, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
			originalSinogram.setSpacing(1, 1.0);
	
			sino = new Grid1D(new float[maxSIndex]);
	
			projector = new ParallelProjectorRayDrivenCL(maxTheta, deltaTheta, maxS, deltaS);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
			originalSinogram.show("Sino");
			backprojector = new ParallelBackprojectorPixelDrivenCL(originalSinogram,x ,y);
			
			art = new ART(projector, backprojector);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

		break;
		
		case (5)://Parallel Beam with OpenCL with StepSize
//			System.out.println("Parallel Beam Open CL, Step size control, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
			originalSinogram.setSpacing(1, 1.0);
			
			diff2D = new Grid2D(originalSinogram.getWidth(), (int)maxThetaIndex);
			diff2D.setSpacing(1	, 1.0);

			gradientSinogram = new Grid2D(originalSinogram.getWidth(), originalSinogram.getHeight());
			gradientSinogram.setSpacing(1, 1.0);
	
			sino = new Grid1D(new float[maxSIndex]);
	
			projector = new ParallelProjectorRayDrivenCL(maxTheta, deltaTheta, maxS, deltaS);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
			originalSinogram.show("Sino");
			backprojector = new ParallelBackprojectorPixelDrivenCL(originalSinogram,x ,y);			
	
			art = new ARTStepSizeControl(projector, backprojector, diff2D, gradientSinogram);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

		break;
		
		case (6)://Fan Beam with OpenCL
//			System.out.println("Fan Beam OpenCL, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(maxTIndex, maxBetaIndex);	
//			originalSinogram = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
			originalSinogram.setSpacing(deltaT, deltaBeta);
	
			sino = new Grid1D(new float[maxTIndex]);
	
			projector = new FanProjectorRayDrivenCL(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
//			originalSinogram.show("Original Sinogram");
			backprojector = (Backprojector) new FanBackprojectorPixelDrivenCL(focalLength, deltaT, deltaBeta, x, y, maxT, maxTIndex, maxBeta, maxBetaIndex);
	
			art = new ART(projector, backprojector);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

		break;
		
		case (7)://Fan Beam with OpenCL with StepSizeControl
//			System.out.println("Fan Beam OpenCL, Step size control, Subsets: " + numberOfSubsets);
			originalSinogram = new Grid2D(maxTIndex, maxBetaIndex);
			originalSinogram.setSpacing(deltaT, deltaBeta);

			diff2D = new Grid2D(originalSinogram.getWidth(), (int)maxBetaIndex);
			diff2D.setSpacing(deltaT, deltaBeta);

			gradientSinogram = new Grid2D(originalSinogram.getWidth(), originalSinogram.getHeight());
			gradientSinogram.setSpacing(deltaT, deltaBeta);
		
			sino = new Grid1D(new float[maxTIndex]);
		
			projector = new FanProjectorRayDrivenCL(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			originalSinogram = (Grid2D) projector.project(phan, originalSinogram);
//			originalSinogram.show("Sino");
			backprojector = (Backprojector) new FanBackprojectorPixelDrivenCL(focalLength, deltaT, deltaBeta, x, y, maxT, maxTIndex, maxBeta, maxBetaIndex);	
		
			art = new ARTStepSizeControl(projector, backprojector, diff2D, gradientSinogram);
			art.setIterationEnvironment(maxIter, epsilon);
			art.setNumberOfSubsets(numberOfSubsets);
			System.out.println("Case: " + type + " - " + projector.getClass().toString().substring(47) + " - " + art.getClass().toString().substring(47) +  " -  Subsets: " + numberOfSubsets);
			recon = (Grid2D) art.reconstruct(originalSinogram, recon, imageUpdate, localImageUpdate, sino);

		break;
		}
			
//	time = System.nanoTime() - time;
		long time = System.currentTimeMillis()-startTime;

	System.out.println("time:" + time + "s");
	double errorPhan = art.calculateError(phan, recon);
	System.out.println("Error phan "+ errorPhan);

	}
}

