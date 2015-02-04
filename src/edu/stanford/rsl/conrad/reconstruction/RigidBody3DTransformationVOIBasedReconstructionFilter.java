package edu.stanford.rsl.conrad.reconstruction;


import ij.process.FloatProcessor;

import java.util.ArrayList;

import Jama.Matrix;
import edu.stanford.rsl.apps.gui.pointselector.PointSelector;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.XmlUtils;


/**
 * The VOIBasedReconstructionFilter is an implementation of the backprojection which employs a volume-of-interest (VOI) to
 * speed up reconstruction. Only voxels within the VOI will be regarded in the backprojection step. Often this can save up to 30 to 40 % in computation time
 * as volumes are usually described as boxes but the VOI is just a cylinder.
 * 
 * This version of the reconstruction algorithm applies the motion field stored in 4D_SPLINE_LOCATION before the backprojection.
 * 
 * @author J.-H. Choi
 *
 */
public class RigidBody3DTransformationVOIBasedReconstructionFilter extends VOIBasedReconstructionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4449313613390154787L;	

	private ArrayList<Matrix> TranslationTransform;// = new ArrayList<Matrix>(getGeometry().getNumProjectionMatrices());
	private ArrayList<Matrix> RotationTransform;// = new ArrayList<Matrix>(getGeometry().getNumProjectionMatrices());

	private String rotationTranslationFilename = null;

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();		
	}

	protected synchronized void initialize(Grid2D projection){
		if (!init){
			super.initialize(projection);		
			readInTRMatrix();			
		}
	}

	public void backproject(Grid2D projection, int projectionNumber){
		int count = 0;
		//System.out.println(projectionVolume);
		if ((!init)){
			initialize(projection);
		}

		FloatProcessor currentProjection = new FloatProcessor(projection.getWidth(), projection.getHeight(), projection.getBuffer(), null);
		//ImageProcessor currentProjection = projection;
		int p = projectionNumber;
		double[] voxel = new double [4];
		SimpleMatrix mat = getGeometry().getProjectionMatrix(p).computeP();
		SimpleVector centerTranlation = null;

		voxel[3] = 1;
		double [] times = new double [getGeometry().getNumProjectionMatrices()];
		for (int i=0; i< times.length; i++){
			times [i]= ((double)i) / getGeometry().getNumProjectionMatrices();
		}
		System.out.println("Processing projection " + p);
		if (mat != null){
			boolean nanHappened = false;
			for (int k = 0; k < maxK ; k++){ // for all slices
				//			for (int k = 150; k < 250 ; k=k+50){ // for all slices
				if (debug) System.out.println("here: " + " " + k);
				voxel[2] = (this.getGeometry().getVoxelSpacingZ() * (k)) - offsetZ;
				for (int j = 0; j < maxJ; j++){ // for all voxels
					voxel[1] = (this.getGeometry().getVoxelSpacingY() * j) - offsetY;
					for (int i=0; i < maxI; i++){ // for all lines
						voxel[0] = (this.getGeometry().getVoxelSpacingX() * i) - offsetX;
						// compute real world coordinates in homogeneous coordinates;
						boolean project = true;
						if (useVOImap){
							if (voiMap != null){
								project = voiMap[i][j][k];
							}
						}
						if (project){			
							PointND point0 = new PointND(voxel[0], voxel[1], voxel[2]);
							PointND point1 = new PointND(voxel[0], voxel[1], voxel[2]); // will be transformed
							if (centerTranlation !=null){
								point1.getAbstractVector().add(centerTranlation);
							}

							// compute compensated position in 3D
							point1 = getTransformedPoint(p, point1);

							if (centerTranlation != null){
								point1.getAbstractVector().subtract(centerTranlation);
							}

							// Compute coordinates in projection data.
							SimpleVector homogeneousPoint1 = SimpleOperators.multiply(mat, new SimpleVector(point1.get(0), point1.get(1), point1.get(2), voxel[3]));
							// Transform to 2D coordinates
							double coordX = homogeneousPoint1.getElement(0)/homogeneousPoint1.getElement(2);
							double coordY = homogeneousPoint1.getElement(1)/homogeneousPoint1.getElement(2);

							// Get Distance weighting L, use point0 (before transformed) in order that uniform distance weighting otherwise low freq noise will be intruded. 
							SimpleVector homogeneousPoint0 = SimpleOperators.multiply(mat, new SimpleVector(point0.get(0), point0.get(1), point0.get(2), voxel[3]));

							// back project 
							double increment = currentProjection.getInterpolatedValue(coordX + lineOffset, coordY) / Math.pow(homogeneousPoint0.getElement(2), 2);
							//							double increment = currentProjection.getInterpolatedValue(coordX + lineOffset, coordY) / Math.pow(homogeneousPoint1.getElement(2), 2);
							if (Double.isNaN(increment)){
								nanHappened = true;
								if (count < 10) System.out.println("NAN Happened at i = " + i + " j = " + j + " k = " + k + " projection = " + projectionNumber + " x = " + coordX + " y = " + coordY  );
								increment = 0;
								count ++;
							}
							updateVolume(i, j, k, increment);
						}
					}
				}
			}
			if (nanHappened) {
				throw new RuntimeException("Encountered NaN in projection!");
			}
			if (debug) System.out.println("done with projection");
		}
	}

	private PointND getTransformedPoint(int imageIndex, PointND point) {

		//PointND pointTransformed = new PointND(0, 0, 0);		
		Matrix Tmat = TranslationTransform.get(imageIndex);
		Matrix Rmat = RotationTransform.get(imageIndex);

		double [][] arrayPoint =
			{
				{point.get(0)},
				{point.get(1)},
				{point.get(2)},
			};		

		Matrix Pmat = new Matrix(arrayPoint);				
		Matrix Rlt = Rmat.times(Pmat).plus(Tmat);

		return new PointND(Rlt.get(0, 0), Rlt.get(1, 0), Rlt.get(2, 0));
	}

	public void readInTRMatrix() {

		TranslationTransform = new ArrayList<Matrix>(getGeometry().getNumProjectionMatrices());
		RotationTransform = new ArrayList<Matrix>(getGeometry().getNumProjectionMatrices());

		// load data from XML file
		ArrayList<double[][][]> RotTrans = null;
		try {
			if (rotationTranslationFilename == null)
				rotationTranslationFilename = FileUtil.myFileChoose(".xml", false);
			
			RotTrans = (ArrayList<double[][][]>)XmlUtils.importFromXML(rotationTranslationFilename);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// bring data into matrix format
		for (int i=0; i< getGeometry().getNumProjectionMatrices(); i++){	
			RotationTransform.add(i, new Matrix(RotTrans.get(0)[i]));
			TranslationTransform.add(i, new Matrix(RotTrans.get(1)[i]));
		}

		
		// Code to export the double[][][] arrays to XML
		/*
		ArrayList<double[][][]> rotTrans = new ArrayList<double[][][]>(2);
		rotTrans.add(arrayRotationTransform);
		rotTrans.add(arrayTranslationTransform);

		try {
			XmlUtils.exportToXML(rotTrans);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 */

	}


	@Override
	public String getName() {
		return "3D Rigid-body Transformation CPU-based Backprojector";
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}


	@Override
	public String getToolName() {
		return "3D Rigid-body Transformation Motion-compensated VOI-based Backprojector";
	}

	// uncomment the save to xml part in readInTRMatrix() and use this main to call the method
	// IMPORTANT: make the method public beforehand
	/*
	public static void main(String[] args) {
		CONRAD.setup();
		RigidBody3DTransformationVOIBasedReconstructionFilter filter = 
				new RigidBody3DTransformationVOIBasedReconstructionFilter();
		filter.readInTRMatrix();
	}
	*/

}
/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/