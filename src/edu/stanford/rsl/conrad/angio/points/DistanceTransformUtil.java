package edu.stanford.rsl.conrad.angio.points;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.util.image.ImageOps;

public class DistanceTransformUtil {

	
	public static Grid3D slicewiseDistanceTransform(Grid3D img, ArrayList<Skeleton> skeletons){
		Grid3D distance = new Grid3D(img);
		for(int k = 0; k < img.getSize()[2]; k++){
			System.out.println("Distance transform on slice "+String.valueOf(k+1)+" of "+String.valueOf(img.getSize()[2]));
			Grid2D vt = SkeletonUtil.skelToBinaryImg(img.getSubGrid(k), skeletons.get(k));
			// calculate final distance map for reconstruction, optimal points have 0 value
			ArrayList<PointND> skel = ImageOps.thresholdedPointList(vt, 0.5);
			DistanceTransform2D distTrafo = new DistanceTransform2D(vt, skel, true);
			vt = distTrafo.run();
			distance.setSubGrid(k, vt);
		}
		return distance;
	}
	
}
