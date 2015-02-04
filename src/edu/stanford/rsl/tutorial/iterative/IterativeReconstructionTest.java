package edu.stanford.rsl.tutorial.iterative;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.RamLakKernel;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.Phantom3D;
import edu.stanford.rsl.tutorial.phantoms.Sphere3D;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

/**
 * A test class for the computation and display of iterative reconstructions.
 * 
 * @author Mario Amrehn
 * 
 */
public class IterativeReconstructionTest {

	private enum MyPhantom {PSPHERE, PSHEPP};

	// -----------------------------------------
	private final static boolean USE_CL_FP = true;	// GPU acceleration
	private final static boolean USE_CL_BP = true;	// GPU acceleration
	
	private final static float	sartRelax 		= 0.8f;
	private final static int	sartIterations 	= 100;
	private final static MyPhantom phan 		= MyPhantom.PSHEPP; // PSHEPP, PSPHERE

	private final static float 	omega 			= 0.8f;
	private final static int 	gdIter 			= 10;
	private final static float 	regul 			= (float) Math.pow(10, -4);
	private final static float 	initStepsize	= 0.3f;

	private final static boolean computeFDKReco		= true;
	private final static boolean computeSARTCLReco	= true;
	private final static boolean computeSARTCPUReco	= false;
	private final static boolean computeETVReco		= true;
	// -----------------------------------------
	
	public static void main(String[] args) {
		
		new ImageJ();

		Configuration.loadConfiguration();

		Configuration conf = Configuration.getGlobalConfiguration();

		Trajectory geo = conf.getGeometry();
		double focalLength = geo.getSourceToDetectorDistance();
		// int maxU = geo.getDetectorWidth();
		// int maxV = geo.getDetectorHeight();
		int maxU_PX = geo.getDetectorWidth();
		int maxV_PX = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		double maxU = (maxU_PX) * deltaU;
		double maxV = (maxV_PX) * deltaV;
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		double imgSpacingX = geo.getVoxelSpacingX();
		double imgSpacingY = geo.getVoxelSpacingY();
		double imgSpacingZ= geo.getVoxelSpacingZ();
		double originX = geo.getOriginX();
		double originY = geo.getOriginY();
		double originZ = geo.getOriginZ();
		
		Grid3D grid = getInput(phan, 
				imgSizeX, imgSizeY, imgSizeZ, 
				imgSpacingX, imgSpacingY, imgSpacingZ,
				originX, originY, originZ);
		grid.show("object");
		
		NumericGridOperator gop = grid.getGridOperator();

		long timeFBP=1, timeSARTCL=-1, timeSARTCPU=-1, timeETV=-1;
		Grid3D recImageFBP=null, recImageIterSartCL=null, recImageIterSartCPU=null, recImageIterETV=null;
		if (computeFDKReco) {
			timeFBP = System.currentTimeMillis();
			recImageFBP = reconstructFBP(grid, focalLength, maxU, maxV, deltaU,
					deltaV, maxU_PX, maxV_PX, conf, geo);
			timeFBP = System.currentTimeMillis() - timeFBP;
			recImageFBP.show("recImageFBP, Min: " + gop.min(recImageFBP) + " Max: " + gop.max(recImageFBP));
		}
		
		if (computeSARTCLReco) {
			timeSARTCL = System.currentTimeMillis();
			recImageIterSartCL = reconstructSART(grid, sartRelax, sartIterations, true);
			timeSARTCL = System.currentTimeMillis() - timeSARTCL;
			recImageIterSartCL.show("recImageIterSART_CL");
		}
		
		if (computeSARTCPUReco) {
			timeSARTCPU = System.currentTimeMillis();
			recImageIterSartCPU = reconstructSART(grid, sartRelax, sartIterations, false);
			timeSARTCPU = System.currentTimeMillis() - timeSARTCPU;
			recImageIterSartCPU.show("recImageIterSART_CPU");
		}
		
		if (computeETVReco) {
			timeETV = System.currentTimeMillis();
			recImageIterETV = reconstructETV(grid, sartRelax, sartIterations,
					omega, gdIter, regul, initStepsize);
			timeETV = System.currentTimeMillis() - timeETV;
			recImageIterETV.show("recImageIterETV");
		}

		System.out
				.format("Config (Phantom %s):\n SART: Relax %.2f, Iter %d\n GD: Iter %d, Regul %.2e, Step %.2f\n Omega %.2f\n",
						phan, sartRelax, sartIterations, gdIter, regul,
						initStepsize, omega);

		System.out.println("Errors:");
		double rmseFBP = -1, rmseSART_CL = -1, rmseSART_CPU = -1, rmseETV = -1;
		if (computeFDKReco) {
			rmseFBP = gop.rmse(recImageFBP, grid);
			System.out.format(" RMSE(GT,FBP): %.7f\n", rmseFBP);
		}
		if (computeSARTCLReco) {
			rmseSART_CL = gop.rmse(recImageIterSartCL, grid);
			System.out.format(" RMSE(GT,SART_CL): %.7f\n", rmseSART_CL);
		}
		if (computeSARTCPUReco) {
			rmseSART_CPU = gop.rmse(recImageIterSartCPU, grid);
			System.out.format(" RMSE(GT,SART_CPU): %.7f\n", rmseSART_CPU);
		}
		if (computeETVReco) {
			rmseETV = gop.rmse(recImageIterETV, grid);
			System.out.format(" RMSE(GT,eTV): %.7f\n", rmseETV);
		}
		
		System.out.println("Norms:");
		double l1GT = -1, l1FBP = -1, l1SART_CL = -1, l1SART_CPU = -1, l1ETV = -1;
		l1GT = gop.normL1(grid);
		System.out.format(" l1(GT): %.2f\n", l1GT);
		if (computeFDKReco) {
			l1FBP = gop.normL1(recImageFBP);
			System.out.format(" l1(FBP): %.2f\n", l1FBP);
		}
		if (computeSARTCLReco) {
			l1SART_CL = gop.normL1(recImageIterSartCL);
			System.out.format(" l1(SART_CL): %.2f\n", l1SART_CL);
		}
		if (computeSARTCPUReco) {
			l1SART_CPU = gop.normL1(recImageIterSartCPU);
			System.out.format(" l1(SART_CPU): %.2f\n", l1SART_CPU);
		}
		if (computeETVReco) {
			l1ETV = gop.normL1(recImageIterETV);
			System.out.format(" l1(eTV): %.2f\n", l1ETV);
		}
		
		System.out.println("Time:");
		if (computeFDKReco)
			System.out.format(" time(FBP): %.1f seconds\n", ((double) timeFBP) / 1000.0);
		if (computeSARTCLReco)
			System.out.format(" time(SART_CL): %.1f%%\n", ((double) timeSARTCL * 100)/ timeFBP);
		if (computeSARTCPUReco)
			System.out.format(" time(SART_CPU): %.1f%%\n", ((double) timeSARTCPU * 100) / timeFBP);
		if (computeETVReco)
			System.out.format(" time(eTV): %.1f%%\n", ((double) timeETV * 100) / timeFBP);

		/*
		if (computeFDKReco && computeSARTCPUReco && computeSARTCLReco) {
			System.out
					.format("Norms:\n l1(GT): %.2f\n l1(FBP): %.2f\n l1(SART_CL): %.2f\n l1(SART:CPU): %.2f\n",
							GridOp.l1Norm(grid), GridOp.l1Norm(recImageFBP),
							GridOp.l1Norm(recImageIterSartCL),
							GridOp.l1Norm(recImageIterSartCPU));

			System.out.format(
					"Recon time: FBP: %.1f seconds. SART_CL: %.1f%%, SART_CPU: %.1f%%",
					((double) timeFBP) / 1000.0, ((double) timeSARTCL * 100)
							/ timeFBP, ((double) timeSARTCPU * 100) / timeFBP);
		}
		
		if (computeETVReco && computeFDKReco && computeSARTCPUReco) {
			System.out
					.format("Norms:\n l1(GT): %.2f\n l1(FBP): %.2f\n l1(SART): %.2f\n l1(eTV): %.2f\n",
							GridOp.l1Norm(grid), GridOp.l1Norm(recImageFBP),
							GridOp.l1Norm(recImageIterSartCPU),
							GridOp.l1Norm(recImageIterETV));

			System.out.format(
					"Recon time: FBP: %.1f seconds. SART: %.1f%%, eTV: %.1f%%",
					((double) timeFBP) / 1000.0, ((double) timeSARTCPU * 100)
							/ timeFBP, ((double) timeETV * 100) / timeFBP);
		}
		*/
		
		// Config (Phantom PSPHERE):
		// SART: Relax 0,80, Iter 400
		// GD: Iter 40, Regul 1,00e-04, Step 0,30
		// Omega 0,80
		// Errors:
		// RMSE(GT,FBP): 0,2315102
		// RMSE(GT,SART): 0,3176485
		// RMSE(GT,eTV): 0,3176485
		// Norms:
		// l1(GT): 389,00
		// l1(FBP): 354,79
		// l1(SART): 346,35
		// l1(eTV): 346,35
		// Recon time: FBP: 3,3 seconds. SART: 13085,2%, eTV: 13409,9%

		// if (true)
		// return;
	}

	private static Grid3D getInput(MyPhantom phan, 
			int imgSizeX, int imgSizeY, int imgSizeZ, 
			double imgSpacingX, double imgSpacingY, double imgSpacingZ,
			double originX, double originY, double originZ) {
		Grid3D res = null;
		switch (phan) {
		case PSPHERE:
			Phantom3D test3D = new Sphere3D(imgSizeX, imgSizeY, imgSizeZ);
			res = test3D;
			break;
		case PSHEPP:
			NumericalSheppLogan3D shepp3d = new NumericalSheppLogan3D(imgSizeX,
					imgSizeY, imgSizeZ);
			res = shepp3d.getNumericalSheppLoganPhantom();
			break;
		default:
			res = new Sphere3D(imgSizeX, imgSizeY, imgSizeZ);
		}
		if(null != res)
			res.setSpacing(imgSpacingX, imgSpacingY, imgSpacingZ);
			res.setOrigin(originX, originY, originZ);
		return res;
	}

	private static Grid3D reconstructETV(Grid3D grid, float sartRelax,
			int eTvIerations, float omega, int gdIter, float regul,
			float initStepsize) {
		ConeBeamProjector cbp = new ConeBeamProjector();
		Grid3D sino = USE_CL_FP ? cbp.projectRayDrivenCL(grid) : cbp
				.projectPixelDriven(grid);
		Etv reconEtv = null;
		try {
			reconEtv = new Etv(grid.getSize(), grid.getSpacing(),
					grid.getOrigin(), sino, sartRelax, omega, gdIter, regul,
					initStepsize);
			reconEtv.iterateETV(eTvIerations);
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (null == reconEtv) {
			System.err.println("Error creating eTV reconstruction instance");
			return null;
		}
		Grid3D recImageIterEtv = reconEtv.getVol();
		return recImageIterEtv;
	}

	private static Grid3D reconstructSART(Grid3D grid, float sartRelax,
			int sartIterations, boolean USE_CL_SART) {
		ConeBeamProjector cbp = new ConeBeamProjector();
		Grid3D sino = USE_CL_FP ? cbp.projectRayDrivenCL(grid) : cbp
				.projectPixelDriven(grid);
		sino.show("sinoCL-SART");
		Sart reconSart = null;
		try {
			if(USE_CL_SART)
				reconSart = new SartCL(grid.getSize(), grid.getSpacing(),
						grid.getOrigin(), sino, sartRelax);
			else
			reconSart = new SartCPU(grid.getSize(), grid.getSpacing(),
					grid.getOrigin(), sino, sartRelax);
			reconSart.iterate(sartIterations);
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (null == reconSart) {
			System.err.println("Error creating SART reconstruction instance");
			return null;
		}
		Grid3D recImageIterSart = reconSart.getVol();
		return recImageIterSart;
	}

	/*
	private static Grid3D reconstructFBPNew(Grid3D grid, double focalLength,
			double maxU, double maxV, double deltaU, double deltaV,
			int maxU_PX, int maxV_PX, Configuration conf, Trajectory geo) {

		ConeBeamProjector cbp = new ConeBeamProjector();
		Grid3D sino = USE_CL_FP ? cbp.projectRayDrivenCL(grid) : cbp
				.projectPixelDriven(grid);
		String sinoType = USE_CL_FP ? "CL" : "CPU";
		sino.show("sino" + sinoType);

		ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength,
				maxU, maxV, deltaU, deltaV);
		RamLakKernel ramKRampFilter = new RamLakKernel(maxU_PX, deltaU);
		int stacksize = conf.getGeometry().getProjectionStackSize();
		double D = conf.getGeometry().getSourceToDetectorDistance();
		int numProjMatrices = geo.getNumProjectionMatrices();
		float factor = (float) (D * D * Math.PI / numProjMatrices);
		for (int i = 0; i < stacksize; ++i) {
			cbFilter.applyToGrid(sino.getSubGrid(i));
			for (int j = 0; j < maxV_PX; ++j)
				ramKRampFilter.applyToGrid(sino.getSubGrid(i).getSubGrid(j));
			NumericalPointwiseOperators.multiplyBy(sino.getSubGrid(i), factor);
		}
		sino.show("FBP:sinoFiltered");

		ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
		Grid3D recImage = USE_CL_BP ? cbbp.backprojectPixelDrivenCL(sino) : cbbp
				.backprojectPixelDriven(sino);
		return recImage;
	}
	*/
	
	private static Grid3D reconstructFBP(Grid3D grid, double focalLength,
			double maxU, double maxV, double deltaU, double deltaV,
			int maxU_PX, int maxV_PX, Configuration conf, Trajectory geo) {

		ConeBeamProjector cbp = new ConeBeamProjector();
		Grid3D sino = USE_CL_FP ? cbp.projectRayDrivenCL(grid) : cbp
				.projectPixelDriven(grid);
		sino.show("sinoCL");

		ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength,
				maxV, maxU, deltaV, deltaU);
		//cbFilter.show();
		RamLakKernel ramK = new RamLakKernel(maxU_PX, deltaU);
		for (int i = 0; i < geo.getProjectionStackSize(); ++i) {
			cbFilter.applyToGrid(sino.getSubGrid(i));
			// ramp
			for (int j = 0; j < maxV_PX; ++j)
				ramK.applyToGrid(sino.getSubGrid(i).getSubGrid(j));
			float D = (float) geo.getSourceToDetectorDistance();
			NumericPointwiseOperators.multiplyBy(sino.getSubGrid(i), (float) (D * D
					* Math.PI / geo.getNumProjectionMatrices()));
		}
		//sino.show("sinoFilt");

		ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
		Grid3D recImage = USE_CL_BP ? cbbp.backprojectPixelDrivenCL(sino) : cbbp
				.backprojectPixelDriven(sino);
		return recImage;
	}
}
