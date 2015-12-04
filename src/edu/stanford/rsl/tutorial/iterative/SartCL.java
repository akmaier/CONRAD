package edu.stanford.rsl.tutorial.iterative;

import ij.ImageJ;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.HelicalTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;

public class SartCL implements Sart{

	//debug
	boolean debug = false;

	// controll structures
	protected CLContext context = null;

	//geometry
	protected Trajectory geo = null;
	protected int maxProjs;
	protected int width;
	protected int height;
	protected int imgSize[];
	protected double spacing[];
	protected double origin[];
	private long starttime = 0;

	//grids
	protected OpenCLGrid3D volCL;
	private OpenCLGrid3D updBP; 
	private OpenCLGrid2D sinoCL;
	protected OpenCLGrid3D normSino;
	protected OpenCLGrid3D normGrid;
	protected OpenCLGrid3D oProj = null;	
	private OpenCLGrid2D[] oProjP;
	private OpenCLGrid2D[] normSinoP;
	protected CLCommandQueue queueIterative;

	// memory for fast calculations
	protected CLBuffer<FloatBuffer> projMatrices = null;
	protected CLBuffer<FloatBuffer> gInvARmatrix = null;
	protected CLBuffer<FloatBuffer> gSrcPoint = null;

	// buffer for 3D volume:
	protected Grid3D volumeResult;

	//sart variables
	protected final float beta;
	protected float normFactor;

	//projectors
	protected ConeBeamProjector cbp;
	protected ConeBeamBackprojector cbbp;

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
	public SartCL(int[] volDims, double[] spacing, double[] origin, OpenCLGrid3D oProj, float beta) throws Exception {
		if (null == oProj) {
			throw new Exception("SART: No projection data given");
		}
		if (1 > volDims[0] || 1 > volDims[1] || 1 > volDims[2]) {
			throw new Exception(
					"SART: Span of each dimension in the volume has to be a natural number");
		}
		// create initial volume filled with zeros
		initCLDatastructure();
		volCL = new OpenCLGrid3D(new Grid3D(volDims[0], volDims[1], volDims[2]));
		volCL.setOrigin(origin);
		volCL.setSpacing(spacing);

		cbp = new ConeBeamProjector();
		cbbp = new ConeBeamBackprojector();

		this.normSino = createNormProj();
		this.normGrid = createNormGrid();
		
		this.oProj = new OpenCLGrid3D(oProj);
		this.oProj.getDelegate().prepareForDeviceOperation();
		queueIterative.putCopyBuffer(oProj.getDelegate().getCLBuffer(), this.oProj.getDelegate().getCLBuffer()).finish();
		/* calculated once for speedup */
		this.normFactor = (float) (this.oProj.getGridOperator().normL1(this.oProj) / normSino.getGridOperator().normL1(normSino));

		this.beta = beta;
		
		sinoCL = new OpenCLGrid2D(new Grid2D(width,height));
		sinoCL.setSpacing(geo.getPixelDimensionX(),geo.getPixelDimensionY());
		
		updBP = new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		updBP.setOrigin(-origin[0], -origin[1], -origin[2]);
		updBP.setSpacing(spacing[0], spacing[1], spacing[2]);

		oProjP = new OpenCLGrid2D[maxProjs];
		normSinoP = new OpenCLGrid2D[maxProjs];

		for (int j= 0; j< maxProjs; ++j){
			oProjP[j] = new OpenCLGrid2D(oProj.getSubGrid(j));
			normSinoP[j] 	= new OpenCLGrid2D(normSino.getSubGrid(j));
		}
	}

	public SartCL(Grid3D initialVol, Grid3D sino, float beta) throws Exception {
		if (null == initialVol) {
			throw new Exception("SART: No initial volume given");
		}
		if (null == sino) {
			throw new Exception("SART: No projection data given");
		}

		initCLDatastructure();

		volCL = new OpenCLGrid3D(initialVol);
		volCL.setOrigin(initialVol.getOrigin());
		volCL.setSpacing(initialVol.getSpacing());
		cbp = new ConeBeamProjector();
		cbbp = new ConeBeamBackprojector();

		this.normSino = createNormProj();
		this.normGrid = createNormGrid();
		
		this.oProj = new OpenCLGrid3D(sino);

		/* calculated once for speedup*/
		this.beta = beta;
		
		sinoCL = new OpenCLGrid2D(new Grid2D(width,height));
		sinoCL.setSpacing(geo.getPixelDimensionX(),geo.getPixelDimensionY());
		updBP = new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		updBP.setOrigin(-origin[0], -origin[1], -origin[2]);
		updBP.setSpacing(spacing[0], spacing[1], spacing[2]);

		oProjP = new OpenCLGrid2D[maxProjs];
		normSinoP = new OpenCLGrid2D[maxProjs];

		for (int j= 0; j< maxProjs; ++j){
			oProjP[j] = new OpenCLGrid2D(oProj.getSubGrid(j));
			normSinoP[j] = new OpenCLGrid2D(normSino.getSubGrid(j));
		}
		
	}

	/**
	 * @return Normalized projection data
	 * @throws Exception 
	 */
	protected OpenCLGrid3D createNormProj() throws Exception {

		OpenCLGrid3D onesVol = new OpenCLGrid3D(new Grid3D(volCL.getSize()[0],volCL.getSize()[1],volCL.getSize()[2]));
		onesVol.getGridOperator().fill(onesVol, 1);
		OpenCLGrid3D sino = new OpenCLGrid3D(new Grid3D(width,height,maxProjs));

		cbp.fastProjectRayDrivenCL(sino,onesVol);

		onesVol.release();
		// prevent div by zero
		float min = sino.getGridOperator().min(sino);
		//sino.show("Sino of norm projection");
		if (0 >= min)
			sino.getGridOperator().addBy(sino,eps);
		return sino;
	}
	
	protected OpenCLGrid3D createNormGrid() throws Exception {

		OpenCLGrid3D normGrid = new OpenCLGrid3D(new Grid3D(imgSize[0],imgSize[1],imgSize[2]));
		OpenCLGrid2D sinoCLOnes = new OpenCLGrid2D(new Grid2D(width,height));
		sinoCLOnes.getGridOperator().fill(sinoCLOnes, 1.0f);
		
		normGrid.getDelegate().prepareForDeviceOperation();

		cbbp.fastBackprojectPixelDrivenCL(sinoCLOnes,normGrid, 0);

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
			
			boolean[] projIsUsed = new boolean[maxProjs]; // default: false
			int p = 0; // current projection index
	
			for (int n = 0; n < maxProjs; ++n) {
	
				p=n;
				/* edit/init data structures */
				projIsUsed[p] = true;
				
				cbp.fastProjectRayDrivenCL(sinoCL, volCL, p);

				/* update step */
				// NOTE: upd = (oProj - sino) ./ normSino

				sinoCL.getGridOperator().subtractBy(sinoCL, oProjP[p]);
				sinoCL.getGridOperator().divideBySave(sinoCL, normSinoP[p]);
			
				updBP.getGridOperator().fill(updBP, 0);
				cbbp.fastBackprojectPixelDrivenCL(sinoCL,updBP,p);

				// NOTE: vol = vol + updBP * beta	
				updBP.getGridOperator().multiplyBySave(updBP, -beta);
				updBP.getGridOperator().divideBy(updBP, normGrid);
				
				volCL.getGridOperator().addBy(volCL, updBP);
				volCL.getGridOperator().removeNegative(volCL);
		
				/*
				 * Don't use projections with a small angle to each other
				 * subsequently
				 */
				p = (p + maxProjs / 4) % maxProjs;
				for (int ii = 1; projIsUsed[p] && ii < maxProjs; ++ii)
					p = (p + 1) % maxProjs;
			}
			
			if(debug) {
				long endtime = System.currentTimeMillis()-starttime;
				System.out.print("Iter "+i+": ");
				System.out.format("%.3f seconds\n", ((double) endtime) / 1000.0);
				System.out.println(volCL.getGridOperator().normL1(volCL));
				System.out.println("End of Sart Iteration ");
			}
		}
		volCL.getGridOperator().fillInvalidValues(volCL,0);
	}

	public Grid3D getVol(){
		volumeResult = new Grid3D(volCL);
		volumeResult.setOrigin(-geo.getOriginInPixelsX(),-geo.getOriginInPixelsY(),-geo.getOriginInPixelsZ());
		volumeResult.setSpacing(geo.getVoxelSpacingX(),geo.getVoxelSpacingY(),geo.getVoxelSpacingZ());
		return volumeResult;
	}

	private void initCLDatastructure(){
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		geo = conf.getGeometry();
		width =  geo.getDetectorWidth();
		height =  geo.getDetectorHeight();
		maxProjs = geo.getProjectionStackSize();
		// create context
		context = OpenCLUtil.getStaticContext();
		queueIterative = context.getMaxFlopsDevice().createCommandQueue();
		
		//image variables
		imgSize = new int[3];
		origin = new double[3];
		spacing = new double[3];
		imgSize[0] = geo.getReconDimensionX();
		imgSize[1] = geo.getReconDimensionY();
		imgSize[2] = geo.getReconDimensionZ();
		spacing[0] = geo.getVoxelSpacingX();
		spacing[1] = geo.getVoxelSpacingY();
		spacing[2] = geo.getVoxelSpacingZ();
		origin[0] = -geo.getOriginX();
		origin[1] = -geo.getOriginY();
		origin[2] = -geo.getOriginZ();

	}

	public static void main(String[] args) throws Exception {
		boolean helix = false;
		int iterations = 2;
		
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory traj = conf.getGeometry();

		if(helix){
			traj = new HelicalTrajectory(Configuration.getGlobalConfiguration().getGeometry());
		
			// set chosen trajectory
			int stepHel =traj.getNumProjectionMatrices();
			double physicalDetectorHeight = traj.getDetectorHeight()*traj.getPixelDimensionY();
			double stepSize = (physicalDetectorHeight*0.05 / (((double)stepHel)));
			double volumeZSize = physicalDetectorHeight*3*0.05;
			//set chosen trajectory
		
			((HelicalTrajectory) traj).setTrajectory( 	stepHel*3, Configuration.getGlobalConfiguration().getGeometry().getSourceToAxisDistance(), traj.getAverageAngularIncrement(), 
														traj.getDetectorOffsetU(), traj.getDetectorOffsetV(), 
														traj.getDetectorUDirection(), traj.getDetectorVDirection(),
														new SimpleVector(0,0,1), new PointND(0,0,volumeZSize / 2), 0, stepSize);
			conf.setGeometry(traj);
		}
		
		new ImageJ();

		ConeBeamProjector cbp = new ConeBeamProjector();
		OpenCLGrid3D grid = new OpenCLGrid3D(new NumericalSheppLogan3D(traj.getReconDimensionX(),traj.getReconDimensionY(), traj.getReconDimensionZ()).getNumericalSheppLoganPhantom());
		
		grid.setOrigin(-traj.getOriginInPixelsX(),-traj.getOriginInPixelsY(),-traj.getOriginInPixelsZ());
		grid.setSpacing(traj.getVoxelSpacingX(),traj.getVoxelSpacingY(),traj.getVoxelSpacingZ());
		System.out.println("GT: "+grid.getGridOperator().normL1(grid));
		try {
			OpenCLGrid3D sino = new OpenCLGrid3D(new Grid3D(traj.getDetectorWidth(),traj.getDetectorHeight(),traj.getProjectionStackSize()));
			sino.setOrigin(0,0,0);
			sino.setSpacing(1,1,1);
			cbp.fastProjectRayDrivenCL(sino,grid);

			SartCL sart = new SartCL(grid.getSize(),grid.getSpacing(),grid.getOrigin(), sino, 0.8f);
			long start = System.currentTimeMillis();
			sart.iterate(iterations);
			long ende = System.currentTimeMillis()-start;
			System.out.format("Time Sart: %.5f seconds\n", ((double) ende) / 1000.0);
			Grid3D foo = sart.getVol();
			System.out.println("L1: "+foo.getGridOperator().normL1(foo));
			System.out.println("RMSE: "+foo.getGridOperator().rmse(foo,grid));
			foo.show();

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

