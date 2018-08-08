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

	public static void main(String[] args) {
		// TODO Auto-generated method stub
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
			Grid3D artifactFree = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
			Grid3D artifactFreeStruct = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);		
			artifactFree.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
			artifactFree.setSpacing(1,1,1);

			for (int k=0; k < artifactFree.getSize()[2]; ++k) {
				for (int j=0; j < artifactFree.getSize()[1]; ++j){
					for (int i=0; i < artifactFree.getSize()[0]; ++i){
						PointND newPos = new PointND(artifactFree.indexToPhysical(i, j, k));
						double[] posNoSaccade = image.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
						float valNoSaccade = InterpolationOperators.interpolateLinear(image, posNoSaccade[2], posNoSaccade[0], posNoSaccade[1]);
						float valNoSaccadeStruct = InterpolationOperators.interpolateLinear(imageStruct, posNoSaccade[2], posNoSaccade[0], posNoSaccade[1]);
						artifactFree.setAtIndex(i, j, k, valNoSaccade);
						artifactFreeStruct.setAtIndex(i, j, k, valNoSaccadeStruct);
					}
				}
			}
			artifactFree.show("artifactFree (X-FAST)");
			artifactFreeStruct.show("artifactFreeStruct (X-FAST)");
			artifactFree = null;
			artifactFreeStruct = null;
		}
		// Y-FAST Rotation
		AffineTransform rot = new AffineTransform(Rotations.createBasicYRotationMatrix(Math.PI/180.0 * 5.0), new SimpleVector(0,0,0));
		// Saccade Tranlation:
		Translation translat = new Translation (20,0,-20);
		image.applyTransform(rot);
		image.show("the rotated image");

		// resample to motion artifact volume.
		// image contains the data after head rotation; now we still need to introduce a saccade at position saccade;
		int saccadeIndex = 200;
		Grid3D motionArtifact = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		motionArtifact.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
		motionArtifact.setSpacing(1,1,1);
		for (int k=0; k < motionArtifact.getSize()[2]; ++k) {
			for (int j=0; j < motionArtifact.getSize()[1]; ++j){
				for (int i=0; i < motionArtifact.getSize()[0]; ++i){
					PointND newPos = new PointND(motionArtifact.indexToPhysical(i, j, k));
					// sacade
					if (i > saccadeIndex) newPos.applyTransform(translat);
					double[] pos = image.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
					float val = InterpolationOperators.interpolateLinear(image, pos[2], pos[0], pos[1]);
					if ((Math.abs(i- saccadeIndex) <= 1.5)) val = (float) (20000 + (Math.random() * 45000)); 
					motionArtifact.setAtIndex(i, j, k, val);
				}
			}
		}
		motionArtifact.show("motionArtifact");
		motionArtifact = null;
		image = null;
		
		imageStruct.applyTransform(rot);
		imageStruct.show("the rotated image");
		
		Grid3D motionArtifactStruct = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		motionArtifactStruct.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
		motionArtifactStruct.setSpacing(1,1,1);
		for (int k=0; k < motionArtifactStruct.getSize()[2]; ++k) {
			for (int j=0; j < motionArtifactStruct.getSize()[1]; ++j){
				for (int i=0; i < motionArtifactStruct.getSize()[0]; ++i){
					PointND newPos = new PointND(motionArtifactStruct.indexToPhysical(i, j, k));
					// sacade
					if (i > saccadeIndex) newPos.applyTransform(translat);
					double[] pos = imageStruct.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
					float valStruct = InterpolationOperators.interpolateLinear(imageStruct, pos[2], pos[0], pos[1]);
					motionArtifactStruct.setAtIndex(i, j, k, valStruct);
				}
			}
		}
		
		motionArtifactStruct.show("motionArtifactStruct");

	}

}
