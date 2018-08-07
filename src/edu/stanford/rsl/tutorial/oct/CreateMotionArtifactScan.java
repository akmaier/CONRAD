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
		ImagePlus ip = IJ.openImage("E:\\Andreas\\MotionCorrection\\head_rotation_test\\3762_OD_Mac_12x12mm_800x800_angio_merged.tiff");
		ImagePlus ipStruct = IJ.openImage("E:\\Andreas\\MotionCorrection\\head_rotation_test\\3762_OD_Mac_12x12mm_800x800_intensity_merged.tiff");
		Grid3D image = ImageUtil.wrapImagePlus(ip);
		Grid3D imageStruct = ImageUtil.wrapImagePlus(ipStruct);
		int [] size = image.getSize();
		image.setOrigin(-size[0]/2,-size[1]/2,-size[2]/2);
		AffineTransform rot = new AffineTransform(Rotations.createBasicYRotationMatrix(Math.PI/180.0 * 5.0), new SimpleVector(0,0,0));
		Grid3D clone = new Grid3D(image);
		Grid3D cloneStruct = new Grid3D(imageStruct);
		clone.show("the original image");
		image.applyTransform(rot);
		image.show("the rotated image");
		imageStruct.applyTransform(rot);
		imageStruct.show("the rotated image");
		// resample to motion artifact volume.
		// image contains the data after head rotation; now we still need to introduce a saccade at position saccade;
		int saccadeIndex = 100;
		int [] newImageBounds = {500, 433, 500};
		Grid3D motionArtifact = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		Grid3D motionArtifactStruct = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		Grid3D artifactFree = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		Grid3D artifactFreeStruct = new Grid3D (newImageBounds[0],newImageBounds[1],newImageBounds[2]);
		motionArtifact.setOrigin(-newImageBounds[0]/2,-newImageBounds[1]/2,-newImageBounds[2]/2);
		motionArtifact.setSpacing(1,1,1);
		// Saccade Tranlation:
		Translation translat = new Translation (50,0,-50);
		for (int k=0; k < motionArtifact.getSize()[2]; ++k) {
			for (int j=0; j < motionArtifact.getSize()[1]; ++j){
				for (int i=0; i < motionArtifact.getSize()[0]; ++i){
					PointND newPos = new PointND(motionArtifact.indexToPhysical(i, j, k));
					double[] posNoSaccade = image.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
					float valNoSaccade = InterpolationOperators.interpolateLinear(image, posNoSaccade[2], posNoSaccade[0], posNoSaccade[1]);
					float valNoSaccadeStruct = InterpolationOperators.interpolateLinear(imageStruct, posNoSaccade[2], posNoSaccade[0], posNoSaccade[1]);
					artifactFree.setAtIndex(i, j, k, valNoSaccade);
					artifactFreeStruct.setAtIndex(i, j, k, valNoSaccadeStruct);
					// sacade
					if (i > saccadeIndex) newPos.applyTransform(translat);
					double[] pos = image.physicalToIndex(newPos.get(0),newPos.get(1), newPos.get(2));
					float val = InterpolationOperators.interpolateLinear(image, pos[2], pos[0], pos[1]);
					float valStruct = InterpolationOperators.interpolateLinear(imageStruct, pos[2], pos[0], pos[1]);
					if ((Math.abs(i- saccadeIndex) <= 1.5) && (val > 1800)) val = (float) (Math.random() * 34500); 
					motionArtifact.setAtIndex(i, j, k, val);
					motionArtifactStruct.setAtIndex(i, j, k, valStruct);
				}
			}
		}
		motionArtifact.show("motionArtifact");
		motionArtifactStruct.show("motionArtifactStruct");
		artifactFree.show("artifactFree");
		artifactFreeStruct.show("artifactFreeStruct");
	}

}
