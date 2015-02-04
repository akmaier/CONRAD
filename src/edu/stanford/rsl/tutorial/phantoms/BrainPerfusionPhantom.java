/*
 * Copyright (C) 2014 Michael Manhart, Kerstin Mueller
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.tutorial.phantoms;

import java.io.IOException;
import java.util.ArrayList;

import ij.ImageJ;
import ij.io.FileInfo;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.ConfigFileBasedTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.GridRawIOUtil;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLForwardProjectorDynamicVolume;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.XmlUtils;


/**
 * @author Kerstin Mueller
 * BrainPerfusionPhantom allows to generate noise-free 2D projection images of a brain perfusion scan
 */
public class BrainPerfusionPhantom {
	BrainPerfusionPhantomConfig config;

	// forward and backward sweeps, calibrated with projection matrices
	protected Trajectory gFwd;
	protected Trajectory gBwd;

	// pre-generated data from the MATLAB Script which can be downloaded here: http://www5.cs.fau.de/en/research/data/digital-brain-perfusion-phantom/
	protected Grid3D volSkull = null;
	protected Grid3D volTissue = null;

	// variables for the contrast flow simulation
	protected Grid3D  volContrastPrev = null;
	protected Grid3D  volContrastNext = null;
	protected int     volContrastPrevIdx = -1;
	protected int     volContrastNextIdx = -1;

	OpenCLForwardProjectorDynamicVolume openclFwdProjector;

	protected static NumericGridOperator gop = NumericGridOperator.getInstance();
	
	public static void main(String[] args) {
		CONRAD.setup();

		
		
		// Create new config file for perfusion phantom
		BrainPerfusionPhantomConfig perfConfig = new BrainPerfusionPhantomConfig();
		if (args.length >= 5)
		{
			// the directory containing, skull.raw and all generated volumes with MATLAB
			perfConfig.phantomDirectory = args[0];
			// complete path with filename to projection matrix for forward run
			perfConfig.calibrationFwdMatFile = args[1];
			// complete path with filename to projection matrix for backward run
			perfConfig.calibrationBwdMatFile = args[2];
			// the output directory for all projections
			perfConfig.phantomOutDirectory = args[3];
			// if motion should be applied
			perfConfig.motion = Boolean.parseBoolean(args[4]);
			// sampling of the generated volumes
			perfConfig.phantomSampling = 1.0f;
			if (args.length==6) {
				if (!args[5].isEmpty())
				{
					// If 3D Markers should be added to the skull phantom
					perfConfig.markerFile = args[5];
				}
			}
		}  else {
			System.err.println("Usage -directory of the phantom - fwd projection matrix file [.txt]"
					+ "- bwd projection matrix file [.txt] - output directory -motion[true/false]");
			System.exit(1);
		}

		BrainPerfusionPhantom perfPhantom = new BrainPerfusionPhantom(perfConfig);

		if ( !perfConfig.markerFile.isEmpty())
		{
			perfPhantom.addMarkers();
		}
		
		try {
			perfPhantom.createProjectionData();
		}catch(IOException io){
			System.err.println("PerfusionBrainPhantom.createProjectionData(): IO error!");
			io.printStackTrace();
		}

	}

	/**
	 * Adds 3D markers to the skull volume, which are predefined in an xml file
	 * 
	 */
	public void addMarkers()
	{
		// Add markers
		// load our 3d marker positions
		ArrayList<double[]> threeDMarkerPositions=null;
		try {
			threeDMarkerPositions = (ArrayList<double[]>) XmlUtils.importFromXML(config.markerFile);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("Exception while reading xml file: " + e);
		}	
		
		Grid3D tempSkullVolume = getSkullVolume();
		Grid3D tempTissueVolume = getTissueVolume();
		
		// Create the tantalum markers with the correct attenuation
		float radius_mm = 0.0f;// mm
		Material beads = MaterialsDB.getMaterial("tantalum");
		float attenutation_beads = (float)(beads.getAttenuation(80, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION))/10;
		
		int radius_px = (int) (radius_mm / BrainPerfusionPhantomConfig.voxelSize);
		for ( int i = 0; i < threeDMarkerPositions.size(); i++)
		{
			int indexX = (int)(threeDMarkerPositions.get(i)[0]);
			int indexY = (int)(threeDMarkerPositions.get(i)[1]);
			int indexZ = (int)(threeDMarkerPositions.get(i)[2]);
			
			for (int rx =-radius_px; rx <= radius_px; rx++)
			{
				for (int ry =-radius_px; ry <= radius_px; ry++)
				{
					for (int rz =-radius_px; rz <= radius_px; rz++)
					{
						tempSkullVolume.addAtIndex(indexX+rx, indexY+ry, indexZ+rz, attenutation_beads);
					}
				}
			}
		}
		setSkullVolume(tempSkullVolume);
		gop.addBySave(tempSkullVolume,tempTissueVolume);
		tempSkullVolume.show("Skull with attached markers");
	}
	
	/**
	 * Creates the 2D projection data
	 * @throws IOException
	 */
	public void createProjectionData() throws IOException
	{
		// -----------------------------------------------------
		// material decomposed dynamic forward projection 
		// -----------------------------------------------------		
		dynamicDensityForwardProjection();
	}

	/**
	 * Performs the dynamic density forward projection in forward, backward runs divided for tissue, skull and contrast agent
	 * @throws IOException
	 */
	protected void dynamicDensityForwardProjection() throws IOException
	{
		openclFwdProjector = new OpenCLForwardProjectorDynamicVolume();
		int[] volumeSize  = new int[3];
		float[] voxelSize = new float[3];
		volumeSize[0]=BrainPerfusionPhantomConfig.sizeX;
		volumeSize[1]=BrainPerfusionPhantomConfig.sizeY;
		volumeSize[2]=BrainPerfusionPhantomConfig.sizeZ;
		voxelSize[0]=BrainPerfusionPhantomConfig.voxelSize;
		voxelSize[1]=BrainPerfusionPhantomConfig.voxelSize;
		voxelSize[2]=BrainPerfusionPhantomConfig.voxelSize;
		openclFwdProjector.configure(gFwd, gBwd, volumeSize, voxelSize);

		Grid2D[] projections = new Grid2D[gFwd.getNumProjectionMatrices()];
		// skull // already sets the projection matrices to the graphics card
		setVolumeToProject(volSkull);
		
		float rot = 30.0f;
		float [] trans = {0,0,0};			
		for(int projection_id = 0; projection_id < gFwd.getNumProjectionMatrices(); projection_id++)
		{
			if (config.motion && projection_id > 120 )
			{
				Projection proj = gFwd.getProjectionMatrix(projection_id);
				Projection projMat = applyRigidMotion(proj, rot, trans);
				gFwd.setProjectionMatrix(projection_id, projMat);
			}				
			System.out.println("Forward projection: skull forward rotation projection " + (projection_id+1));	
			projections[projection_id] = applyForwardProjection(projection_id, true, config.motion);
		}
		saveProjectionData("skull_fwd", projections);
		for(int projection_id = 0; projection_id < gBwd.getNumProjectionMatrices(); projection_id++)
		{
			if (config.motion && projection_id > 120 )
			{
				Projection proj = gFwd.getProjectionMatrix(projection_id);
				Projection projMat = applyRigidMotion(proj, rot, trans);
				gFwd.setProjectionMatrix(projection_id, projMat);
			}	
			System.out.println("Forward projection: skull backward rotation projection " + (projection_id+1));
			projections[projection_id] = applyForwardProjection(projection_id, false, config.motion);
		}				
		saveProjectionData("skull_bwd", projections);

		//tissue
		setVolumeToProject(volTissue);
		for(int projection_id = 0; projection_id < gFwd.getNumProjectionMatrices(); projection_id++)
		{
			if (config.motion && projection_id > 120 )
			{
				Projection proj = gFwd.getProjectionMatrix(projection_id);
				Projection projMat = applyRigidMotion(proj, rot, trans);
				gFwd.setProjectionMatrix(projection_id, projMat);
			}	
			System.out.println("Forward projection: tissue forward rotation projection " + (projection_id+1));
			projections[projection_id] = applyForwardProjection(projection_id, true, config.motion);
		}
		saveProjectionData("tissue_fwd", projections);
		for(int projection_id = 0; projection_id < gBwd.getNumProjectionMatrices(); projection_id++)
		{
			if (config.motion && projection_id > 120 )
			{
				Projection proj = gFwd.getProjectionMatrix(projection_id);
				Projection projMat = applyRigidMotion(proj, rot, trans);
				gFwd.setProjectionMatrix(projection_id, projMat);
			}	
			System.out.println("Forward projection: tissue backward rotation projection " + (projection_id+1));
			projections[projection_id] = applyForwardProjection( projection_id, false, config.motion);
		}
		saveProjectionData("tissue_bwd", projections);

		// contrast agent
		double time_increment_projection = config.tRot/(gFwd.getNumProjectionMatrices()-1);
		for(int sweep = 0; sweep < config.numSweeps; sweep++)
		{
			boolean fwd = (sweep%2 == 0);
			float time = config.tStart+sweep*(config.tRot+config.tPause);
			for(int projection_id = 0; projection_id < gFwd.getNumProjectionMatrices(); projection_id++)
			{
				if (config.motion && projection_id > 120 )
				{
					Projection proj = gFwd.getProjectionMatrix(projection_id);
					Projection projMat = applyRigidMotion(proj, rot, trans);
					gFwd.setProjectionMatrix(projection_id, projMat);
				}	
				System.out.println("Forward projection contrast agent: sweep " + (sweep+1) + " projection " + (projection_id+1));
				setVolumeToProject(getDynamicContrastVolume(time));
				projections[projection_id] = applyForwardProjection(projection_id, fwd, config.motion);
				time += time_increment_projection;
			}
			String fn_proj_out = String.format("sweep%03d_contrast", sweep);
			saveProjectionData(fn_proj_out, projections);
		}	

	}

	/**
	 * Applies 3D rotation and translation to the projection matrix 
	 * y = P * [R|t] * x; P(3x4) and R|t (4x4)
	 * @param proj projection where to apply the motion
	 * @param angle_degree
	 * @param translation float[]   
	 * @return projection with changed projection matrix
	 */
	protected Projection applyRigidMotion(Projection proj, float angle_degree, float [] translation)
	{
		double angle_radian = 0.0;
		angle_radian = angle_degree*(Math.PI/180.0);
			
		SimpleMatrix P = proj.computeP();
		SimpleMatrix Motion = new SimpleMatrix(4,4);
		Motion.fill(0.0);
		Motion.setElementValue(0, 0, Math.cos(angle_radian)); // angle in radians
		Motion.setElementValue(0, 1, Math.sin(angle_radian)); // angle in radians
		Motion.setElementValue(1, 0, -Math.sin(angle_radian)); // angle in radians
		Motion.setElementValue(1, 1, Math.cos(angle_radian)); // angle in radians
		Motion.setElementValue(2, 2, 1); 
		Motion.setElementValue(0, 3, translation[0]);
		Motion.setElementValue(1, 3, translation[1]);
		Motion.setElementValue(2, 3, translation[2]);
		Motion.setElementValue(3, 3, 1);
		SimpleMatrix newProj = SimpleOperators.multiplyMatrixProd(P, Motion);
		Projection projNew = new Projection();
		projNew.initFromP(newProj);
		return projNew; 
	}

	/**
	 * Sets the 3D tissue volume for projection 
	 * @param 
	 */
	protected void setTissueVolume(Grid3D tissueVolume)
	{
		volTissue = tissueVolume;
	}
	
	
	/**
	 * Gets the 3D skull volume for projection 
	 * @param 
	 */
	protected Grid3D getTissueVolume()
	{
		return volTissue;
	}
	
	/**
	 * Sets the 3D skull volume for projection 
	 * @param 
	 */
	protected void setSkullVolume(Grid3D skullVolume)
	{
		volSkull = skullVolume;
	}
	
	
	/**
	 * Gets the 3D skull volume for projection 
	 * @param 
	 */
	protected Grid3D getSkullVolume()
	{
		return volSkull;
	}
	
	/**
	 * Sets the 3D volume for projection 
	 * @param volumeBuffer Grid3D containing the information for forward projection
	 */
	protected void setVolumeToProject(Grid3D volumeBuffer)
	{
		float[] linearBuffer = new float[volumeBuffer.getSize()[0]*volumeBuffer.getSize()[1]*volumeBuffer.getSize()[2]];
		int i = 0;
		for(Grid2D grid : volumeBuffer.getBuffer()){
			for(int j = 0; j < grid.getBuffer().length; j++){
				linearBuffer[j+i] = grid.getBuffer()[j];
			}
			i += grid.getBuffer().length;
		}
		openclFwdProjector.setVolume(linearBuffer);
	}

	/**
	 * Performs the forward projection onto a given projection image
	 * @param projectionId the number for the projection image and matrix
	 * @param fwd defines if it is a forward or backward run
	 * @return 2D projection image 
	 */
	protected Grid2D applyForwardProjection(int projectionId, boolean fwd, boolean motion )
	{	
		Trajectory g = fwd?gFwd:gBwd;
		Grid2D projection = new Grid2D(g.getDetectorWidth(),g.getDetectorHeight());
		openclFwdProjector.applyForwardProjection(projectionId, fwd, projection.getBuffer(), motion);
		return projection;	
	}

	/**
	 * Saves the projection image stack to file
	 * @param filename the name of the file (note: not complete path!)
	 * @param grid contains an array of all projection images
	 */
	public void saveProjectionData(String filename, Grid2D[] grid)
	{
		FileInfo fI = GridRawIOUtil.getDefaultFloat32BigEndianFileInfo(grid[0]);
		String sep = System.getProperty("file.separator");
		String complFilename = config.phantomOutDirectory+sep+filename;
		try {
			GridRawIOUtil.saveRawDataGrid(grid, fI, complFilename);
		} catch (IOException e) {
			System.err.println("Could not write projection data to file "+ filename);
			e.printStackTrace();
		}
	}

	/**
	 * Computes the dynamic contrasted volume inside the brain tissue
	 * @param time the time point in s 
	 * @return the 3D volume of the contrasted brain
	 * @throws IOException
	 */
	protected Grid3D getDynamicContrastVolume(float time) throws IOException
	{
		Grid3D volume = new Grid3D(BrainPerfusionPhantomConfig.sizeX,BrainPerfusionPhantomConfig.sizeY,BrainPerfusionPhantomConfig.sizeZ);
		FileInfo fI = GridRawIOUtil.getDefaultFloat32BigEndianFileInfo();
		fI.nImages = BrainPerfusionPhantomConfig.sizeZ;

		int prev_idx = (int)(time/config.phantomSampling)+1;
		float prev_time = (float)((prev_idx-1)*config.phantomSampling);
		int next_idx = (int)(time/config.phantomSampling)+2;
		float next_time = (float)((next_idx-1)*config.phantomSampling);

		String sep = System.getProperty("file.separator");

		if(prev_idx != volContrastPrevIdx) {
			String complFilename = config.phantomDirectory+sep+Integer.toString(prev_idx);
			volContrastPrev = (Grid3D) GridRawIOUtil.loadFromRawData(fI, complFilename);
			HUToAttenuationValues(volContrastPrev);
			volContrastPrevIdx = prev_idx;
		}
		if(next_idx != volContrastNextIdx) {
			String complFilename = config.phantomDirectory+sep+Integer.toString(next_idx);
			volContrastNext = (Grid3D) GridRawIOUtil.loadFromRawData(fI, complFilename);
			HUToAttenuationValues(volContrastNext);
			volContrastNextIdx = next_idx;
		}

		float weight = (float)(time-prev_time)/(next_time-prev_time);

		NumericPointwiseIteratorND pIter = new NumericPointwiseIteratorND(volume);
		NumericPointwiseIteratorND pIterNext = new NumericPointwiseIteratorND(volContrastNext);
		NumericPointwiseIteratorND pIterPrev = new NumericPointwiseIteratorND(volContrastPrev);

		while( pIter.hasNext() && pIterPrev.hasNext() && pIterNext.hasNext())
		{
			float perfusionValue = (1-weight)*pIterPrev.getNext() + weight*pIterNext.getNext();
			pIter.setNext(perfusionValue);
		}
		return volume;
	}

	
	public void HUToAttenuationValues(Grid3D volume)
	{
		NumericPointwiseIteratorND pIter = new NumericPointwiseIteratorND(volume);
		while (pIter.hasNext())
		{
			// :1000 + 1: scale from HU values to density (0 HU = 1 mg/mm^3; voxel volume is 1 mm^3)
			pIter.setNext(pIter.get()/1000 + 1);
		}
	}
	
	/** 
	 * Reads the skull file generated with the MATLAB Tool (crate_phantom_material_decomposition.m)
	 */
	public void readSkullInfo()  
	{
		FileInfo fi = GridRawIOUtil.getDefaultFloat32BigEndianFileInfo();
		fi.width = BrainPerfusionPhantomConfig.sizeX;
		fi.height = BrainPerfusionPhantomConfig.sizeY;
		fi.nImages = BrainPerfusionPhantomConfig.sizeZ;
		String sep = System.getProperty("file.separator");
		String complFilename = config.phantomDirectory+sep+"skull";
		volSkull = (Grid3D) GridRawIOUtil.loadFromRawData(fi, complFilename);
		HUToAttenuationValues(volSkull);
		
	}

	/**
	 * Reads the tissue file generated with the MATLAB Tool (crate_phantom_material_decomposition.m)
	 */
	public void readTissueInfo()  
	{
		FileInfo fi = GridRawIOUtil.getDefaultFloat32BigEndianFileInfo();
		fi.width = BrainPerfusionPhantomConfig.sizeX;
		fi.height = BrainPerfusionPhantomConfig.sizeY;
		fi.nImages = BrainPerfusionPhantomConfig.sizeZ;
		String sep = System.getProperty("file.separator");
		String complFilename = config.phantomDirectory+sep+"tissue";
		volTissue = (Grid3D) GridRawIOUtil.loadFromRawData(fi, complFilename);
		HUToAttenuationValues(volTissue);
	}

	/**
	 * Constructor
	 * @param config contains all information regarding the phantom generation process
	 */
	public BrainPerfusionPhantom(BrainPerfusionPhantomConfig config)
	{
		this.config = config;

		gFwd = ConfigFileBasedTrajectory.openAsGeometrySource(config.calibrationFwdMatFile, Configuration.getGlobalConfiguration().getGeometry());
		gBwd = ConfigFileBasedTrajectory.openAsGeometrySource(config.calibrationFwdMatFile, Configuration.getGlobalConfiguration().getGeometry());


		readSkullInfo();
		readTissueInfo();
		if ( volSkull==null || volTissue==null) 
		{
			System.err.println("BrainPerfusionPhantom: Error reading phantom input data!");
		}
		new ImageJ();
		
	}

}