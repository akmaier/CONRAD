package edu.stanford.rsl.tutorial.iterative;

import ij.ImageJ;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLContext;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom3D;
import edu.stanford.rsl.tutorial.phantoms.Sphere3D;

/*
 * TODO: implement parrallel beam reoncstruction
 */
public class SartCL2D{

	//debug
	boolean debug = false;

	// controll structures
	protected CLContext context = null;
	
	//geometry
	protected enum reconGeometry {PARALLEL, FAN};
	protected static reconGeometry usedGeometry = reconGeometry.FAN;

	//geometry
	protected Trajectory geo = null;
	protected int width;
	protected int height;
	int imgSizeX;
	int imgSizeY;
	double spacingX;
	double spacingY;
	double originX;
	double originY;
	double focalLength;
	double maxBeta;
	double deltaBeta;
	double maxT;
	double deltaT;
	int maxBetaIndex;
	int maxTIndex;
	long starttime = 0;

	//grids
	protected OpenCLGrid2D volCL;
	private OpenCLGrid2D updBP; 
	private OpenCLGrid2D sinoCL;
	protected OpenCLGrid2D normSino;
	protected OpenCLGrid2D normGrid;
	protected OpenCLGrid2D oProj = null;	

	// memory for fast calculations
	protected CLBuffer<FloatBuffer> projMatrices = null;
	protected CLBuffer<FloatBuffer> gInvARmatrix = null;
	protected CLBuffer<FloatBuffer> gSrcPoint = null;

	// buffer for 3D volume:
	protected Grid2D volumeResult;

	//sart variables
	protected final float beta;
	protected float normFactor;

	//projectors
	protected FanBeamProjector2D fbp;
	protected FanBeamBackprojector2D fbbp;
	protected ParallelProjector2D pbp;
	protected ParallelBackprojector2D pbbp;
	
	float eps = (float)1.0e-10;

	/**
	 * This constructor takes the following arguments:
	 * @param volDims
	 * @param spacing
	 * @param origin
	 * @param oProj
	 * @param beta
	 * @throws Exception
	 */
	public SartCL2D(int[] volDims, double[] spacing, double[] origin, Grid2D oProj, float beta) throws Exception {
		if (null == oProj) {
			throw new Exception("SART: No projection data given");
		}
		if (1 > volDims[0] || 1 > volDims[1]) {
			throw new Exception(
					"SART: Span of each dimension in the volume has to be a natural number");
		}
		// create initial volume filled with zeros
		initCLDatastructure();
		volCL = new OpenCLGrid2D(new Grid2D(volDims[0], volDims[1]));
		volCL.setOrigin(origin);
		volCL.setSpacing(spacing);
		
		if (usedGeometry == reconGeometry.FAN){
			fbp =  new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			fbbp = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSizeX, imgSizeY);
		} else {
			pbp = new ParallelProjector2D(maxBeta, deltaBeta, maxT, deltaT);
			//pbbp = new ParallelBackprojector2D(imgSizeX,imgSizeY,pxSzXMM,pxSzYMM);
		}

		this.normSino = createNormProj();
		this.normGrid = createNormGrid();
		
		this.oProj = new OpenCLGrid2D(oProj);
		/* calculated once for speedup */
		this.normFactor = (float) (this.oProj.getGridOperator().normL1(this.oProj) / normSino.getGridOperator().normL1(normSino));

		this.beta = beta;
		
		sinoCL = new OpenCLGrid2D(new Grid2D(maxTIndex,maxBetaIndex));
		sinoCL.setSpacing(deltaT, deltaBeta);
		updBP = new OpenCLGrid2D(new Grid2D(imgSizeX,imgSizeY));
		updBP.setOrigin(volCL.getOrigin()[0], volCL.getOrigin()[1]);
		updBP.setSpacing(volCL.getSpacing()[0], volCL.getSpacing()[1]);
	
	}

	public SartCL2D(Grid2D initialVol, Grid2D sino, float beta) throws Exception {
		if (null == initialVol) {
			throw new Exception("SART: No initial volume given");
		}
		if (null == sino) {
			throw new Exception("SART: No projection data given");
		}

		initCLDatastructure();

		volCL = new OpenCLGrid2D(initialVol);

		if (usedGeometry == reconGeometry.FAN){
			fbp =  new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			fbbp = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSizeX, imgSizeY);
		} else {
			pbp = new ParallelProjector2D(maxBeta, deltaBeta, maxT, deltaT);
			//pbbp = new ParallelBackprojector2D(imgSzXMM,imgSzYMM,pxSzXMM,pxSzYMM);
		}

		this.normSino = createNormProj();
		this.normGrid = createNormGrid();
		
		this.oProj = new OpenCLGrid2D(sino);

		/* calculated once for speedup*/
		this.beta = beta;
		
		sinoCL = new OpenCLGrid2D(new Grid2D(maxTIndex,maxBetaIndex));
		sinoCL.setSpacing(deltaT, deltaBeta);
		updBP = new OpenCLGrid2D(new Grid2D(imgSizeX,imgSizeY));
		updBP.setOrigin(-originX, -originY);
		updBP.setSpacing(spacingX, spacingY);
	}

	/**
	 * @return Normalized projection data
	 * @throws Exception 
	 */
	protected OpenCLGrid2D createNormProj() throws Exception {

		OpenCLGrid2D onesVol = new OpenCLGrid2D(new Grid2D(volCL.getSize()[0],volCL.getSize()[1]));
		onesVol.getGridOperator().fill(onesVol, 1);

		OpenCLGrid2D sino = new OpenCLGrid2D(new Grid2D(maxTIndex,maxBetaIndex));

		if (usedGeometry == reconGeometry.FAN){
			fbp.fastProjectRayDrivenCL(sino,onesVol);
		} else {
			//pbp.fastProjectRayDrivenCL(sino,onesVol);
		}
			
		onesVol.release();
		// prevent div by zero
		float min = sino.getGridOperator().min(sino);
		//sino.show("Sino of norm projection");
		if (0 >= min)
			sino.getGridOperator().addBy(sino,eps);
			
		sino.setSpacing(deltaT, deltaBeta);
		return sino;
	}
	
	protected OpenCLGrid2D createNormGrid() throws Exception {

		OpenCLGrid2D normGrid = new OpenCLGrid2D(new Grid2D(imgSizeX,imgSizeY));
		OpenCLGrid2D sinoCLOnes = new OpenCLGrid2D(new Grid2D(maxTIndex,maxBetaIndex));
		sinoCLOnes.getGridOperator().fill(sinoCLOnes, 1.0f);
		sinoCLOnes.setSpacing(deltaT, deltaBeta);
		normGrid.getDelegate().prepareForDeviceOperation();
		if (usedGeometry == reconGeometry.FAN){
			fbbp.fastBackprojectPixelDrivenCL(sinoCLOnes,normGrid);
		} else {
			//pbbp.fastBackProjectRayDrivenCL(sinoCLOnes,normGrid, 0);
		}
		
		normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));
		sinoCLOnes.release();
		
		return normGrid;
	}

	public final void iterate() throws Exception{
		iterate(1);
	}

	public final void iterate(final int iter) throws Exception {
	
		for (int i = 0; i < iter; ++i) {
			if(debug) {
				System.out.println("Starting Sart Iteration "+i);
				starttime = System.currentTimeMillis();
			}

			if (usedGeometry == reconGeometry.FAN){
				fbp.fastProjectRayDrivenCL(sinoCL, volCL);
			} else {
				//pbp.fastProjectRayDrivenCL(sinoCL, volCL, p);
			}

			/* update step */
			// NOTE: upd = (oProj - sino) ./ normSino

			sinoCL.getGridOperator().subtractBy(sinoCL, oProj);
			sinoCL.getGridOperator().divideBySave(sinoCL, normSino);
			
			updBP.getGridOperator().fill(updBP, 0);
			
			if (usedGeometry == reconGeometry.FAN){
				fbbp.fastBackprojectPixelDrivenCL(sinoCL, updBP);
			} else {
				//pbbp.fastBackprojectPixelDrivenCL(sinoCL, updBP, p);
			}

			// NOTE: vol = vol + updBP * beta	
			updBP.getGridOperator().multiplyBySave(updBP, -beta);
			updBP.getGridOperator().divideBy(updBP, normGrid);
			
			volCL.getGridOperator().addBy(volCL, updBP);
			volCL.getGridOperator().removeNegative(volCL);
			}
			if(debug) {
				long endtime = System.currentTimeMillis()-starttime;
				System.out.format("%.3f seconds\n", ((double) endtime) / 1000.0);
				System.out.println("End of Sart Iteration ");
			}

		volCL.getGridOperator().fillInvalidValues(volCL,0);
	}

	public Grid2D getVol2D(){
		volumeResult = new Grid2D(volCL);
		volumeResult.setOrigin(-geo.getOriginInPixelsX(),-geo.getOriginInPixelsY());
		volumeResult.setSpacing(geo.getVoxelSpacingX(),geo.getVoxelSpacingY());
		return volumeResult;
	}

	private void initCLDatastructure(){
		Configuration conf = Configuration.getGlobalConfiguration();
		geo = conf.getGeometry();
		width =  geo.getDetectorWidth();
		// create context
		context = OpenCLUtil.getStaticContext();
		imgSizeX = geo.getReconDimensionX();
		imgSizeY = geo.getReconDimensionY();
		spacingX = geo.getVoxelSpacingX();
		spacingY = geo.getVoxelSpacingY();
		originX = -geo.getOriginX();
		originY = -geo.getOriginY();
		
		focalLength = geo.getSourceToDetectorDistance();
		maxBeta = geo.getAverageAngularIncrement()*geo.getProjectionStackSize()*Math.PI/180.0;
		deltaBeta = maxBeta/geo.getProjectionStackSize();
		maxT = geo.getDetectorWidth();
		deltaT = geo.getPixelDimensionX();
		maxBetaIndex = (int) (maxBeta / deltaBeta);
		maxTIndex = (int) (maxT / deltaT);
		
		height = maxTIndex;
	}

	public static void main(String[] args) {
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		CircularTrajectory traj = new CircularTrajectory(conf.getGeometry());
		((CircularTrajectory) traj).setTrajectory( 36, Configuration.getGlobalConfiguration().getGeometry().getSourceToAxisDistance(), 10.0, 0, 0, 
				CameraAxisDirection.DETECTORMOTION_PLUS, 
				CameraAxisDirection.ROTATIONAXIS_PLUS, 
				new SimpleVector(0,0,1));
		conf.setGeometry(traj);
		new ImageJ();
		
		double focalLength = traj.getSourceToDetectorDistance();
		double maxBeta = traj.getAverageAngularIncrement()*traj.getProjectionStackSize()*Math.PI/180.0;
		double deltaBeta = maxBeta/traj.getProjectionStackSize();
		double maxT = traj.getDetectorWidth();
		double deltaT = traj.getPixelDimensionX();
		int maxBetaIndex = (int) (maxBeta / deltaBeta);
		int maxTIndex = (int) (maxT / deltaT);
		
		FanBeamProjector2D fbp = new FanBeamProjector2D(focalLength,maxBeta,deltaBeta,maxT,deltaT);

		MickeyMouseGrid2D test2D = new MickeyMouseGrid2D(traj.getReconDimensionX(), traj.getReconDimensionY());
		OpenCLGrid2D grid = new OpenCLGrid2D(test2D);
	
		grid.setSpacing(traj.getVoxelSpacingX(),traj.getVoxelSpacingY());
		grid.setOrigin(-(traj.getReconDimensionX() * grid.getSpacing()[0]) / 2, -(traj.getReconDimensionY()* grid.getSpacing()[1]) / 2);
		
		System.out.println("GT: "+grid.getGridOperator().normL1(grid));
		try {
			OpenCLGrid2D sino = new OpenCLGrid2D(new Grid2D(maxTIndex,maxBetaIndex));
			fbp.fastProjectRayDrivenCL(sino,grid);
			
			SartCL2D sart = new SartCL2D(grid.getSize(), grid.getSpacing(), grid.getOrigin(), sino, 0.8f);
			long start = System.currentTimeMillis();
			sart.iterate(100);
			long ende = System.currentTimeMillis()-start;
			System.out.format("Time Sart: %.5f seconds\n", ((double) ende) / 1000.0);
			
			Grid2D foo = sart.getVol2D();
			System.out.println("L1: "+foo.getGridOperator().normL1(foo));
			foo.show();

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

