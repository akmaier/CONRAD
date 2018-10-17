package edu.stanford.rsl.tutorial.oct;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

/**
 * Creates an artificial saccade with head rotation.
 * @author akmaier
 *
 */
public class CreateMotionArtifactScan {
	
	static class Saccade {
		int location;
		Translation translation;
		public Saccade (int loc, Translation trans) {
			location = loc;
			translation = trans;
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Saccade [] saccadesXFast = {new Saccade(100,  new Translation (5,0,0)), new Saccade(400,  new Translation (0,0,-5))};
		Saccade [] saccadesYFast = {new Saccade(200, new Translation (20,0,-20))};
		
		new ImageJ();
		ImagePlus ip = IJ.openImage("E:\\Andreas\\MotionCorrection\\head_rotation_test\\unthresholded_input\\Angio.tif");
		ImagePlus ipStruct = IJ.openImage("E:\\Andreas\\MotionCorrection\\head_rotation_test\\unthresholded_input\\Struct.tif");
		Grid3D image = ImageUtil.wrapImagePlus(ip);
		//image.show("Angio");
		Grid3D imageStruct = ImageUtil.wrapImagePlus(ipStruct);
		//imageStruct.show("Struct");
		int [] size = image.getSize();
		image.setOrigin(-size[0]/2,-size[1]/2,-size[2]/2);
		image.setSpacing(1,1,1);
		imageStruct.setOrigin(-size[0]/2,-size[1]/2,-size[2]/2);
		imageStruct.setSpacing(1,1,1);
		int [] newImageBounds = {500, 433, 500};
		boolean doXFAST = false;
		if (doXFAST) {
			System.out.println("Running XFAST");
			Grid3D xFastImage = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
			Grid3D xFastStruct = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);		
			xFastImage.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
			xFastImage.setSpacing(1,1,1);

			for (int k=0; k < xFastImage.getSize()[2]; ++k) {
				int currentSaccadeIndex = -100;
				Translation currentSaccadeTranslation = new Translation(0,0,0);
				for (int s = 0; s< saccadesXFast.length; s++) {
					if (k> saccadesXFast[s].location-2) {
						currentSaccadeIndex = saccadesXFast[s].location;
						currentSaccadeTranslation = saccadesXFast[s].translation;
					}
				}
				for (int j=0; j < xFastImage.getSize()[1]; ++j){	
					for (int i=0; i < xFastImage.getSize()[0]; ++i){
						PointND newPos = new PointND(xFastImage.indexToPhysical(i, j, k));
						newPos.applyTransform(currentSaccadeTranslation);
						double[] posSaccade = image.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
						float valSaccade = InterpolationOperators.interpolateLinear(image, posSaccade[2], posSaccade[0], posSaccade[1]);
						float valSaccadeStruct = InterpolationOperators.interpolateLinear(imageStruct, posSaccade[2], posSaccade[0], posSaccade[1]);
						if ((Math.abs(k- currentSaccadeIndex) <= 1.5)) valSaccade = (float) (20000 + (Math.random() * 45000)); 
						xFastImage.setAtIndex(i, j, k, valSaccade);
						xFastStruct.setAtIndex(i, j, k, valSaccadeStruct);
					}
				}
			}
			xFastImage.show("X-FAST Angio");
			xFastStruct.show("X-FAST Struct");
			xFastImage = null;
			xFastStruct = null;
		}
		// Y-FAST Rotation
		AffineTransform rot = new AffineTransform(Rotations.createBasicYRotationMatrix(Math.PI/180.0 * 10.0), new SimpleVector(0,0,0));
		// Saccade Tranlation:
		image.applyTransform(rot);
		image.show("the rotated image");

		// resample to motion artifact volume.
		// image contains the data after head rotation; now we still need to introduce a saccade at position saccade;
		Grid3D yFastImage = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		yFastImage.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
		yFastImage.setSpacing(1,1,1);
		for (int k=0; k < yFastImage.getSize()[2]; ++k) {
			for (int j=0; j < yFastImage.getSize()[1]; ++j){
				for (int i=0; i < yFastImage.getSize()[0]; ++i){
					PointND newPos = new PointND(yFastImage.indexToPhysical(i, j, k));
					// sacade
					int currentSaccadeIndex = -100;
					Translation currentSaccadeTranslation = new Translation(0,0,0);
					for (int s = 0; s< saccadesYFast.length; s++) {
						if (i> saccadesYFast[s].location-2) {
							currentSaccadeIndex = saccadesYFast[s].location;
							currentSaccadeTranslation = saccadesYFast[s].translation;
						}
					}
					newPos.applyTransform(currentSaccadeTranslation);
					double[] pos = image.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
					float val = InterpolationOperators.interpolateLinear(image, pos[2], pos[0], pos[1]);
					if ((Math.abs(i- currentSaccadeIndex) <= 1.5)) val = (float) (20000 + (Math.random() * 45000)); 
					yFastImage.setAtIndex(i, j, k, val);
				}
			}
		}
		yFastImage.show("Y-Fast Angio");
		yFastImage = null;
		image = null;
		
		imageStruct.applyTransform(rot);
		imageStruct.show("the rotated image");
		
		Grid3D yFastStruct = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		yFastStruct.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
		yFastStruct.setSpacing(1,1,1);
		for (int k=0; k < yFastStruct.getSize()[2]; ++k) {
			for (int j=0; j < yFastStruct.getSize()[1]; ++j){
				for (int i=0; i < yFastStruct.getSize()[0]; ++i){
					PointND newPos = new PointND(yFastStruct.indexToPhysical(i, j, k));
					// sacade
					Translation currentSaccadeTranslation = new Translation(0,0,0);
					for (int s = 0; s< saccadesYFast.length; s++) {
						if (i> saccadesYFast[s].location-2) {
							currentSaccadeTranslation = saccadesYFast[s].translation;
						}
					}
					newPos.applyTransform(currentSaccadeTranslation);
					double[] pos = imageStruct.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
					float valStruct = InterpolationOperators.interpolateLinear(imageStruct, pos[2], pos[0], pos[1]);
					yFastStruct.setAtIndex(i, j, k, valStruct);
				}
			}
		}
		
		yFastStruct.show("motionArtifactStruct");

	}

}
