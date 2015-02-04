package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

public class RayDrivenBasedReconstruction extends IterativeReconstruction {
	

	private static final long serialVersionUID = 1L;
	
	protected Grid3D projectionImage;
	protected int lineOffset = 0;

	protected final int maxU = getGeometry().getDetectorWidth(); //or it should be projection.getWidth();
	protected final int maxV = getGeometry().getDetectorHeight();

	protected final int maxK = getGeometry().getReconDimensionZ();
	protected final int maxI = getGeometry().getReconDimensionX();
	protected final int maxJ = getGeometry().getReconDimensionY();
	
	protected final double dx = getGeometry().getVoxelSpacingX();
	protected final double dy = getGeometry().getVoxelSpacingY();
	protected final double dz = getGeometry().getVoxelSpacingZ();

	int umin;
	int umax;
	int vmin; 
	int vmax;
	double []weights;
	SimpleMatrix mat;
	
	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		init = false;
	}

	public synchronized void Initialize(){
		if (!init){
			super.init();
			//configuration?????
			
			InitializeProjectionImage();
			Grid2D projection =  inputQueue.get(0);
			
			//if projectionWidth does not equal to detectorWidth
			//this needs to be further discussed.
			if (getGeometry().getDetectorWidth() != -1){
				System.out.println("row size projection: " + projection.getWidth() + "\nrow size detector: " + getGeometry().getDetectorWidth());
				lineOffset = (projection.getWidth() - getGeometry().getDetectorWidth()) / 2;  //have not considered if the lineOffset is not integer 
			}

			copyProjectionImage();

		}
	}
	
   public void InitializeProjectionImage() {
	   	projectionImage = new Grid3D(nImages,maxU,maxV);
		if (debug) System.out.println("Created Projection Image");
		computeOffsets();
	}
	
	public void copyProjectionImage(){
		Grid2D projection;
		for ( int ip = 0; ip < nImages; ip++ ){
			try {
				projection = inputQueue.get(ip);
				for (int iu = 0; iu <= maxU ; iu ++ ){
					for (int iv = 0; iv <= maxV ; iv++){
						// there may be a problem
						projectionImage.setAtIndex(ip, iu, iv, projection.getPixelValue(iu + lineOffset, iv));
							
					}
				}	
			} catch (Exception e){
				System.out.println("An error occured during backprojection of projection " + ip);
			}
		}

	}

	public void backproject() throws Exception {
		for ( int p = 0; p < nImages; p++ ){
			mat = getGeometry().getProjectionMatrix(p).computeP();
			
			for( int k = 0; k < maxK; k++){
				for( int i = 0; i < maxI; i++ ){
					for( int j = 0; j < maxJ; j++){
						
						if ( getSystemMatrixEntries( i, j, k) ){
							//Add values to projectionImage
						}
					}
				}	
			}
		}

	}

	
	public void forwardproject() throws Exception {
		for ( int p = 0; p < nImages; p++ ){
			mat = getGeometry().getProjectionMatrix(p).computeP();
			
			for( int k = 0; k < maxK; k++){
				for( int i = 0; i < maxI; i++ ){
					for( int j = 0; j < maxJ; j++){
						
						if ( getSystemMatrixEntries( i, j, k) ){
							//Add values to projectionImage
						}
						
					}
				}	
			}
		}
	} 
	
	public void getSystemMatrixRow(){
		//TODO 
	}
	
	public void getSystemMatrixColumn(){
		//TODO
	}
 
	// column/voxel index -> list of pixel indexes with weights
	public boolean getSystemMatrixEntries( int i, int j, int k){
		double[] voxel = new double [4];
		double[] homogeniousPointi = new double[3];
		double[] homogeniousPointj = new double[3];
		double[] homogeniousPointk = new double[3];
		double[][] updateMatrix = new double [3][4];
		
		if (mat != null){
			updateMatrix[0][3] = mat.getElement(0,3);
			updateMatrix[1][3] = mat.getElement(1,3);
			updateMatrix[2][3] = mat.getElement(2,3);
			voxel[3] = 1;
			
			voxel[0] = (this.getGeometry().getVoxelSpacingX() * i) - offsetX;
			updateMatrix[0][0] = mat.getElement(0,0) * voxel[0];
			updateMatrix[1][0] = mat.getElement(1,0) * voxel[0];
			updateMatrix[2][0] = mat.getElement(2,0) * voxel[0];
			homogeniousPointi[0] = homogeniousPointk[0] + updateMatrix[0][0];
			homogeniousPointi[1] = homogeniousPointk[1] + updateMatrix[1][0];
			homogeniousPointi[2] = homogeniousPointk[2] + updateMatrix[2][0];
			
			voxel[1] = (this.getGeometry().getVoxelSpacingY() * j) - offsetY;
			updateMatrix[0][1] = mat.getElement(0,1) * voxel[1];
			updateMatrix[1][1] = mat.getElement(1,1) * voxel[1];
			updateMatrix[2][1] = mat.getElement(2,1) * voxel[1];
			homogeniousPointj[0] = homogeniousPointi[0] + updateMatrix[0][1];
			homogeniousPointj[1] = homogeniousPointi[1] + updateMatrix[1][1];
			homogeniousPointj[2] = homogeniousPointi[2] + updateMatrix[2][1];
			
			voxel[2] = (this.getGeometry().getVoxelSpacingZ() * (k)) - offsetZ;
			updateMatrix[0][2] = mat.getElement(0,2) * voxel[2];
			updateMatrix[1][2] = mat.getElement(1,2) * voxel[2];
			updateMatrix[2][2] = mat.getElement(2,2) * voxel[2];
			homogeniousPointk[0] = updateMatrix[0][3] + updateMatrix[0][2];
			homogeniousPointk[1] = updateMatrix[1][3] + updateMatrix[1][2];
			homogeniousPointk[2] = updateMatrix[2][3] + updateMatrix[2][2];
			
			double coordX = homogeniousPointj[0] / homogeniousPointj[2];
			double coordY = homogeniousPointj[1] / homogeniousPointj[2];

			umin = (int) Math.floor (coordX);
			umax = (int) Math.ceil  (coordX);
			vmin = (int) Math.floor (coordY);
			vmax = (int) Math.ceil  (coordY);
			
			umin = Math.max(umin, maxU);
			umax = Math.min(umax, 0);
			vmin = Math.max(vmin, maxV);
			vmax = Math.min(vmax, 0);
			
			if ( umax > umin && vmax > vmin){
				return false;
			}
			
			//compute the weights
		}
		return true;		
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

	@Override
	public String getToolName() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void iterativeReconstruct() throws Exception {
		// TODO Auto-generated method stub
		
	}

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/