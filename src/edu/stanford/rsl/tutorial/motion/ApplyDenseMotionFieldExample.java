package edu.stanford.rsl.tutorial.motion;

import java.io.IOException;

import org.itk.simple.doubleArray;
import org.itk.simple.int16Array;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import ij.ImagePlus;
import edu.stanford.rsl.apps.gui.XCatMetricPhantomCreator;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.geometry.motion.OpenCLParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.motion.ParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class ApplyDenseMotionFieldExample {

	public static void main(String[] args) {
		int timesteps = 10; 
		CONRAD.setup();
		Configuration config = Configuration.getGlobalConfiguration();
		XCatMetricPhantomCreator creator = new XCatMetricPhantomCreator();
		creator.setSteps(timesteps);
		AnalyticPhantom4D scene = creator.instantiateScene();
		ImagePlus hyper = creator.renderMetricVolumePhantom(scene);
		Grid3D hyperGrid = ImageUtil.wrapImagePlus(hyper);
		hyperGrid.show("Sampled from Phantom");
		Grid3D deformed = new Grid3D(hyperGrid, false);
		float motionfield [] [] = new float [timesteps][];
		PointND splineSurfacePoints [] [] = new PointND [timesteps][];
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		ParzenWindowMotionField motion = (ParzenWindowMotionField) scene.getMotionField();
		OpenCLParzenWindowMotionField cl;
		int factor = 16; 
		double originx = config.getGeometry().getOriginX();
		try {
			cl = new OpenCLParzenWindowMotionField(motion, context, device);
			for (int i = 0; i < timesteps; i++){
				double time = ((double)i)/(timesteps-1);
				motionfield[i] =cl.getMotionFieldAsArrayReduceZGridXY(time, 0, config.getGeometry().getReconDimensionX()/factor, config.getGeometry().getReconDimensionY()/factor); 	
				splineSurfacePoints [i] = cl.getRasterPoints(time);
			}
			
			for (int z = 0; z < deformed.getSize()[2]; z++){
				MultiChannelGrid2D slice = new MultiChannelGrid2D(deformed.getSize()[0], deformed.getSize()[1], timesteps);
				for (int i = 0; i < timesteps; i++){
					Grid2D slicet = new Grid2D(deformed.getSize()[0], deformed.getSize()[1]);
					double zCoord = edu.stanford.rsl.conrad.geometry.General.voxelToWorld(z, config.getGeometry().getVoxelSpacingZ(), config.getGeometry().getOriginZ());
					int idxz = z*(deformed.getSize()[0]*deformed.getSize()[1]);
					for (int y = 0; y < deformed.getSize()[1]; y++){
						double yCoord = edu.stanford.rsl.conrad.geometry.General.voxelToWorld(y, config.getGeometry().getVoxelSpacingY(), config.getGeometry().getOriginY());
						int idxy = y*(deformed.getSize()[0]);
						for (int x = 0; x < deformed.getSize()[0]; x++){
							double xCoord = edu.stanford.rsl.conrad.geometry.General.voxelToWorld(x, config.getGeometry().getVoxelSpacingX(), originx);
							int idx = idxz + idxy + x;
							double xCoordTemp = xCoord + (motionfield[i][(idx*3)]);
							double yCoordTemp = yCoord + (motionfield[i][(idx*3)+1]);
							double zCoordTemp = zCoord + (motionfield[i][(idx*3)+2]);
							double xCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(xCoordTemp, config.getGeometry().getVoxelSpacingX(), originx);
							double yCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(yCoordTemp, config.getGeometry().getVoxelSpacingY(), config.getGeometry().getOriginY());
							double zCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(zCoordTemp, config.getGeometry().getVoxelSpacingZ(), config.getGeometry().getOriginZ());
							float value = InterpolationOperators.interpolateLinear(hyperGrid, zCoordVoxel, xCoordVoxel, yCoordVoxel);
							slicet.setAtIndex(x, y, value);
						}
					}
					slice.setChannel(i, slicet);
				}
				deformed.setSubGrid(z, slice);
			}
			for (int i = 0; i < timesteps; i++){
				for (int j = 0; j < splineSurfacePoints[0].length; j++){
					PointND point = splineSurfacePoints [0][j].clone();
					PointND from = splineSurfacePoints [i][j].clone();
					
					double xCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(from.get(0), config.getGeometry().getVoxelSpacingX(), originx);
					double yCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(from.get(1), config.getGeometry().getVoxelSpacingY(), config.getGeometry().getOriginY());
					double zCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(from.get(2), config.getGeometry().getVoxelSpacingZ(), config.getGeometry().getOriginZ());
					int z = (int) Math.round(zCoordVoxel);
					int y = (int) Math.round(yCoordVoxel);
					int x = (int) Math.round(xCoordVoxel);
					int idx = z*(deformed.getSize()[0]*deformed.getSize()[1]) + y*(deformed.getSize()[0]) + x;
					
					double xCoordTemp = from.get(0) + (motionfield[i][(idx*3)]);
					double yCoordTemp = from.get(1) + (motionfield[i][(idx*3)+1]);
					double zCoordTemp = from.get(2) + (motionfield[i][(idx*3)+2]);
					PointND interp = new PointND(xCoordTemp, yCoordTemp, zCoordTemp);
					
					xCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(xCoordTemp, config.getGeometry().getVoxelSpacingX(), originx);
					yCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(yCoordTemp, config.getGeometry().getVoxelSpacingY(), config.getGeometry().getOriginY());
					zCoordVoxel = edu.stanford.rsl.conrad.geometry.General.worldToVoxel(zCoordTemp, config.getGeometry().getVoxelSpacingZ(), config.getGeometry().getOriginZ());
					//z = (int) Math.round(zCoordVoxel);
					//y = (int) Math.round(yCoordVoxel);
					//x = (int) Math.round(xCoordVoxel);
					
					MultiChannelGrid2D multi = (MultiChannelGrid2D) deformed.getSubGrid(z);
					if (multi !=null) {
						//multi.getChannel(i).setAtIndex(x, y, j);
					}
					point.getAbstractVector().subtract(from.getAbstractVector());
					interp.getAbstractVector().subtract(from.getAbstractVector());
					SimpleVector diff = interp.getAbstractVector().clone();
					diff.subtract(point.getAbstractVector());
					if (Math.abs(diff.normL2()) > 0.01) {
						//System.out.println(j+": error: " + diff.normL2() + " expected point:" + point + " interp:" +interp);
					}
				}
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		deformed.show("Phantom created with deformation");
	}

}
