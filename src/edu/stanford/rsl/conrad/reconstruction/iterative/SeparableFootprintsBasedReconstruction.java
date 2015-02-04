package edu.stanford.rsl.conrad.reconstruction.iterative;

import java.lang.Math;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


public class SeparableFootprintsBasedReconstruction extends	ModelBasedIterativeReconstruction {


	private static final long serialVersionUID = 1L;
	public boolean Debug = true;
	public boolean Debug1 = false;
	public boolean Debug2 = false;
	protected static final int MAX_WEIGHT_LENGTH = 6;
	
	protected enum SFOperation{ SF_PROJECT, SF_BACKPROJECT};
	public enum FootprintsType{ RectRect, TrapRect, TrapTrap };

	public SeparableFootprintsBasedReconstruction( Trajectory dataTrajectory ) {
		super(dataTrajectory);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public void iterativeReconstruct() throws Exception {
		
		forwardproject( projectionViews, volumeImage );
	}
	
	@Override
	protected void forwardproject(Grid3D projImage, Grid3D volImage) throws Exception {
		//zero out whole volume image		
		NumericPointwiseOperators.fill(projImage, 0.0f);
		NumericPointwiseOperators.fill(volImage, 1.0f);
		
		//if (Debug) volImage.printOneSlice(3);
		Projection proj;
		
		for ( int p = 0; p < nImages ; p++ ){
			proj = getGeometry().getProjectionMatrix(p);
			separableFootprintsRectRect( projImage, volImage, proj, p , SFOperation.SF_PROJECT);
		}

		//if (Debug) projImage.printOneSlice(8);
	}


	@Override
	protected void backproject(Grid3D projImage, Grid3D volImage) throws Exception {

		//zero out whole volume image
		NumericPointwiseOperators.fill(volImage, 0.0f);

		//if (Debug) volImage.printOneSlice(3);
		Projection proj;

		for ( int p = 0; p < nImages; p++ ){
			proj = getGeometry().getProjectionMatrix(p);
			separableFootprintsRectRect( projImage, volImage, proj, p , SFOperation.SF_BACKPROJECT);
		}

	}
	
	protected void separableFootprintsRectRect( Grid3D projImage, Grid3D volImage, Projection proj, final int ip, final SFOperation operation  ) throws Exception{
		
		
		SimpleMatrix mat = proj.computeP();
		SimpleVector cameraCenter = proj.computeCameraCenter();

		// buffer have move of voxel
		SimpleVector halfVoxelMoveX = mat.getCol(0);
		halfVoxelMoveX.multiplyBy(dx/2);
		SimpleVector halfVoxelMoveY = mat.getCol(1);
		halfVoxelMoveY.multiplyBy(dy/2);
		SimpleVector fullVoxelMoveZ = mat.getCol(2);
		fullVoxelMoveZ.multiplyBy(dz);

		SimpleVector point3d = new SimpleVector(4);
		point3d.setElementValue(3, 1);

		SimpleVector point2d, point2dMinus, point2dPlus;

		double cx = cameraCenter.getElement(0);
		double cy = cameraCenter.getElement(1);
		double cz = cameraCenter.getElement(2);
		double dsx0, dsy0, dsxy0_sqr;
		double coordLeft, coordRight, coordBottom, coordTop, coordStep;
		double ds0, amplitude;
		SeparableFootprints footprints = new SeparableFootprints();;
		boolean projOnDetector;
		
		for (int i=1; i < maxI; i++){

			point3d.setElementValue(0, i*dx - offsetX  );

			for (int j = 0; j < maxJ; j++){
				
				point3d.setElementValue(1, j*dy - offsetY );
				point3d.setElementValue(2, -dz/2 - offsetZ);
				point2d = SimpleOperators.multiply(mat, point3d);

				dsx0 = Math.abs(point3d.getElement(0) - cx);
				dsy0 = Math.abs(point3d.getElement(1) - cy);
				dsxy0_sqr = dsx0*dsx0 + dsy0*dsy0;

				if ( dsy0 > dsx0){
					ds0 = dsy0;
					point2dMinus = SimpleOperators.subtract(point2d, halfVoxelMoveX);
					point2dPlus = SimpleOperators.add(point2d, halfVoxelMoveX);					
				}else{
					ds0 = dsx0;
					point2dMinus = SimpleOperators.subtract(point2d, halfVoxelMoveY);
					point2dPlus = SimpleOperators.add(point2d, halfVoxelMoveY);
				}

				coordLeft =  point2dMinus.getElement(0) / point2dMinus.getElement(2) + 0.5;
				coordRight =  point2dPlus.getElement(0) / point2dPlus.getElement(2) + 0.5;

				projOnDetector = footprints.rectFootprintWeightU(coordLeft, coordRight, maxU);
				
				if ( !projOnDetector ) 
					continue;
				
				coordBottom = point2d.getElement(1) / point2d.getElement(2) + 0.5;
				point2d.add(fullVoxelMoveZ);
				coordTop = point2d.getElement(1) / point2d.getElement(2) + 0.5;
				coordStep = coordTop - coordBottom;

				for ( int k = 0; k < maxK; k++ ){

					double dsz0 = Math.abs(k*dz-offsetZ-cz); 

					projOnDetector = footprints.rectFootprintWeightV(coordBottom, coordTop, maxV);
					
					if ( !projOnDetector ) 
						continue;
					
					amplitude = Math.sqrt(dsz0*dsz0 + dsxy0_sqr) / ds0;
					
					if ( operation == SFOperation.SF_PROJECT ){
						footprints.footprintsProject(  projImage, volImage, amplitude, i, j, k, ip);
					}else if  ( operation == SFOperation.SF_BACKPROJECT ) {
						footprints.footprintsBackproject(  projImage, volImage, amplitude, i, j, k, ip);
					}else{
						throw new Exception("Wrong SF Operation");
					}
						
					coordBottom = coordTop;
					coordTop = coordTop + coordStep;

				} //k
			}//j		
		}//i
	}
	

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/