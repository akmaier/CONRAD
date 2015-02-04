package edu.stanford.rsl.conrad.geometry;


import ij.process.BinaryProcessor;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * All kinds of geometric routines that are not specific to some geometric object or interact on a set of these.
 * @author Andreas Keil
 */
public abstract class General {

	public static final SimpleVector E_X = new SimpleVector(1.0, 0.0, 0.0);
	public static final SimpleVector E_Y = new SimpleVector(0.0, 1.0, 0.0);
	public static final SimpleVector E_Z = new SimpleVector(0.0, 0.0, 1.0);

	private static boolean normalizeMode = false;

	public static boolean isNormalizeMode(){
		return normalizeMode;
	}

	public static void setNormalizeMode(boolean mode){
		normalizeMode = mode;
	}
	
	/**
	 * Computes the convex hull of the projection of the eight corners of the volume to the projection.
	 * Here we assume that the volume is aligned with the world coordinate system and can be described
	 * by a maximal and a minimal coordinate.
	 * 
	 * Note that the returned int array contains the 4 values of the convex hull in projection domain
	 * as sequence minimum x, maximum x, minimum y, maximum y 
	 * 
	 * @param min the minimal 3D coordinate of the volume
	 * @param max the maximal 3D coordinate of the volume
	 * @param proj the projection matrix
	 * @return the convex hull in projection domain.
	 */
	public static int[] projectVolumeToProjection(PointND min, PointND max, Projection proj){
		// bounds in 2D:
		int [] bounds = new int [4];
		// eight corners of the volume in homogeneous coordinates:
		PointND [] corners = new PointND[8];
		corners[0] = new PointND(min.get(0),min.get(1),min.get(2), 1);
		corners[1] = new PointND(min.get(0),min.get(1),max.get(2), 1);
		corners[2] = new PointND(min.get(0),max.get(1),min.get(2), 1);
		corners[3] = new PointND(min.get(0),max.get(1),max.get(2), 1);
		corners[4] = new PointND(max.get(0),min.get(1),min.get(2), 1);
		corners[5] = new PointND(max.get(0),min.get(1),max.get(2), 1);
		corners[6] = new PointND(max.get(0),max.get(1),min.get(2), 1);
		corners[7] = new PointND(max.get(0),max.get(1),max.get(2), 1);
		SimpleVector projected = new SimpleVector(2);
		@SuppressWarnings("unused")
		double depth = proj.project(corners[0].getAbstractVector(), projected);
// TODO: Check depth value to see if point is actually in front of the camera (>0) or behind it (<0)
//		if (depth <= 0.0) ; // add code for invisible point here
// old code:
//		SimpleMatrix p = proj.computeP();
//		SimpleVector projected = SimpleOperators.multiply(p, corners[0].getAbstractVector());
//		projected.divideBy(projected.getElement(2));
		bounds[0] = (int) projected.getElement(0);
		bounds[1] = (int) projected.getElement(0);
		bounds[2] = (int) projected.getElement(1);
		bounds[3] = (int) projected.getElement(1);
		for (int i = 1; i < 8; i++){
			depth = proj.project(corners[i].getAbstractVector(), projected);
// TODO: Check depth value to see if point is actually in front of the camera (>0) or behind it (<0)
//			if (depth <= 0.0) ; // add code for invisible point here
// old code:
//			projected = SimpleOperators.multiply(p, corners[i].getAbstractVector());
//			projected.divideBy(projected.getElement(2));
// proposed new code for min/max operations:
			bounds[0] = java.lang.Math.min(bounds[0], (int)projected.getElement(0)); // TODO: Maybe better use ceil/floor instead of a simple cast here?
			bounds[1] = java.lang.Math.max(bounds[1], (int)projected.getElement(0)); // TODO: Maybe better use ceil/floor instead of a simple cast here?
			bounds[2] = java.lang.Math.min(bounds[2], (int)projected.getElement(1)); // TODO: Maybe better use ceil/floor instead of a simple cast here?
			bounds[3] = java.lang.Math.max(bounds[3], (int)projected.getElement(1)); // TODO: Maybe better use ceil/floor instead of a simple cast here?
// old code:
//			bounds[0] = (bounds[0] > projected.getElement(0)) ? (int) projected.getElement(0) : bounds[0];
//			bounds[1] = (bounds[1] < projected.getElement(0)) ? (int) projected.getElement(0) : bounds[1];
//			bounds[2] = (bounds[2] > projected.getElement(1)) ? (int) projected.getElement(1) : bounds[2];
//			bounds[3] = (bounds[3] < projected.getElement(1)) ? (int) projected.getElement(1) : bounds[3];
		}
		return bounds;
	}
	

	public static SimpleVector crossProduct(final SimpleVector v1, final SimpleVector v2) {
		// input check
		assert (v1.getLen() == 3) : new IllegalArgumentException("v1 has to be a 3-vector!");
		assert (v2.getLen() == 3) : new IllegalArgumentException("v2 has to be a 3-vector!");

		final SimpleVector result = new SimpleVector(3);
		result.setElementValue(0, v1.getElement(1) * v2.getElement(2) - v1.getElement(2) * v2.getElement(1));
		result.setElementValue(1, v1.getElement(2) * v2.getElement(0) - v1.getElement(0) * v2.getElement(2));
		result.setElementValue(2, v1.getElement(0) * v2.getElement(1) - v1.getElement(1) * v2.getElement(0));
		return result;
	}

	public static boolean areColinear(final SimpleVector v1, final SimpleVector v2, final double delta) {
		return (crossProduct(v1, v2).normL2() < delta);
	}

	/**
	 * Computes the angle between two vectors;
	 * @param a
	 * @param b
	 * @return the smaller (and positive) angle between a and b in radians
	 */
	public static double angle(SimpleVector a, SimpleVector b) {
		final double norma = a.normL2();
		final double normb = b.normL2();
		if (norma == 0 || normb == 0)
			throw new RuntimeException("At least one of the vectors has zero length!");
		else
			return (Math.acos(SimpleOperators.multiplyInnerProd(a, b)/(norma * normb)));
	}
	
	public static double euclideanDistance(final SimpleVector v1, final SimpleVector v2) {
		return SimpleOperators.subtract(v1, v2).normL2();
	}

	public static SimpleVector augmentToHomgeneous(final SimpleVector v) {
		SimpleVector v_hom = new SimpleVector(v.getLen() + 1);
		v_hom.setSubVecValue(0, v);
		v_hom.setElementValue(v.getLen(), 1.0);
		return v_hom;
	}

	public static SimpleVector normalizeFromHomogeneous(final SimpleVector v) {
		SimpleVector v_normalized = v.getSubVec(0, v.getLen()-1);
		assert (v.getElement(v.getLen()-1) > CONRAD.DOUBLE_EPSILON) : new RuntimeException("Cannot de-homogenize a point at infinity!");
		v_normalized.divideBy(v.getElement(v.getLen()-1));
		return v_normalized;
	}

	public static SimpleMatrix createHomAffineMotionMatrix(final SimpleMatrix A, final SimpleVector t) {
		// input checks
		final int d = t.getLen(); 
		assert (d == 2 || d == 3);
		assert (A.getRows() == d && A.getCols() == d);

		final SimpleMatrix result = new SimpleMatrix(d+1, d+1);
		result.setSubMatrixValue(0, 0, A);
		result.setSubColValue(0, d, t);
		result.setElementValue(d, d, 1.0);
		return result;
	}

	public static SimpleMatrix createHomAffineMotionMatrix(final SimpleMatrix A) {
		// input checks
		final int d = A.getRows(); 
		assert (d == 2 || d == 3);
		assert A.getCols() == d;

		final SimpleMatrix result = new SimpleMatrix(d+1, d+1);
		result.setSubMatrixValue(0, 0, A);
		result.setElementValue(d, d, 1.0);
		return result;
	}

	public static SimpleMatrix createHomAffineMotionMatrix(final SimpleVector t) {
		// input checks
		final int d = t.getLen(); 
		assert (d == 2 || d == 3);

		final SimpleMatrix result = new SimpleMatrix(d+1, d+1);
		result.identity();
		result.setSubColValue(0, d, t);
		return result;
	}

	public static double toRadians(double degrees){
		return (degrees / 180.0) * Math.PI;
	}
	
	public static double toDegrees(double radians){
		return (radians * 180.0) / Math.PI;
	}
		
	/**
	 * Convert voxel indexes to world coordinates (in mm), given the spacing and
	 * origin of a volume.
	 * 
	 * The origin in world coordinates is defined as the world coordinate of the
	 * center of the first voxel (the 0/0/0 voxel) in world coordinates (in mm).
	 * Example: To center a volume with 512 voxels and 1.5mm spacing in every
	 * direction around the 0mm/0mm/0mm world coordinate, one has to set all the
	 * origin parameters to -383.25mm (= -512/2*1.5mm + 1/2*1.5mm).
	 * 
	 * @see #worldToVoxel(double[] world, double[] spacing, double[] origin)
	 */
	public static double[] voxelToWorld(int[] voxel, double[] spacing, double[] origin) {
		double[] world = new double[3];
		for (int i = 0; i < 3; ++i)
			world[i] = voxelToWorld(voxel[i],spacing[i], origin[i]);
		return world;
	}


	/**
	 * Computes the origin in world coordinates from pixel coordinates. Example<BR>
	 * General.voxelToWorld(-originInPixelsX, getVoxelSpacingX(), 0)
	 * @param voxel the negative voxel coordinate
	 * @param spacing the spacing
	 * @param origin 0
	 * @return the origin in world coordinates
	 */
	public static double voxelToWorld(double voxel, double spacing, double origin) {
		return voxel*spacing + origin;
	}
	
	
	/**
	 * Convert world coordinates (in mm) to voxel indexes, given the spacing and
	 * origin of a volume.
	 * 
	 * @see #voxelToWorld(int[] voxel, double[] spacing, double[] origin) 
	 */
	public static double[] worldToVoxel(double[] world, double[] spacing, double[] origin) {
		double[] voxel = new double[3];
		for (int i = 0; i < 3; ++i)
			voxel[i] = worldToVoxel(world[i], spacing[i], origin[i]);
		return voxel;
	}
	
	/**
	 * Helper function to convert world coordinates (in mm) to voxel indexes.
	 * Example: <BR>
	 * General.worldToVoxel(-originInWorld.get(0), getVoxelSpacingX(), 0)
	 * 
	 * @see #worldToVoxel(double[] world, double[] spacing, double[] origin)
	 */
	public static double worldToVoxel(double world, double spacing, double origin) {
		return (world - origin) / spacing;
	}
	
	public static void splitHomAffineMotionMatrix(final SimpleMatrix At, final SimpleMatrix A, final SimpleVector t) {
		// check input
		assert (At.getRows() == 4 && At.getCols() == 4 && At.getElement(3, 0) == 0 && At.getElement(3,1) == 0 && At.getElement(3,2) == 0 && At.getElement(3,3) > 10.0*CONRAD.DOUBLE_EPSILON) :
			new IllegalArgumentException("At has to be a homogeneous rigid motion matrix!");

		// check output
		assert (A.getRows() == 3 && A.getCols() == 3) :
			new IllegalArgumentException("A has to be a 3x3 matrix!");
		assert (t.getLen() == 3) :
			new IllegalArgumentException("t has to be a 3-vector!");

		// extract normalized A and t
		final double scale = 1.0/At.getElement(3, 3);
		A.init(At.getSubMatrix(0, 0, 3, 3).multipliedBy(scale));
		t.init(At.getSubCol(0, 3, 3).multipliedBy(scale));
	}

	public static ArrayList<PointND> intersectRayWithCuboid(StraightLine line, PointND min, PointND max){
		double [] vals = new double [2];
		ArrayList<PointND> revan = new ArrayList<PointND>();
		if (General.intersectRayWithCuboid(line.getPoint().getAbstractVector(), line.getDirection(), min.getAbstractVector(), max.getAbstractVector(), vals)){
			revan.add(line.evaluate(vals[0]));
			revan.add(line.evaluate(vals[1]));
		}
		return revan;
	}
	
	/**
	 * Method to check whether a point is within a given cubiod defined by min and max. 
	 * @param point the point
	 * @param min the minimum coordinate of the cuboid
	 * @param max the maximum coordinate of the cuboid.
	 * @return true, if the point is inside
	 */
	public static boolean isWithinCuboid(PointND point, PointND min, PointND max){
		boolean fulfilled = true;
		for (int i =0; i < point.getDimension(); i++){
			fulfilled = fulfilled && (point.get(i)<max.get(i)) && (point.get(i)>min.get(i));
		}
		return fulfilled;
	}
	
	/**
	 * Computes the two intersections of a ray with a cuboid, called entry and
	 * exit point where the ray is specified by the given line origin and ray direction.
	 *
	 * The entry and exit points are returned as distances from the line origin along the
	 * ray. The world coordinates of the entry and exit points may be computed as follows:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   \\mathbf{v}_n & = \\mathbf{C} + t_n \\cdot \\mathbf{d} \\
	 * 	 \\mathbf{v}_f & = \\mathbf{C} + t_f \\cdot \\mathbf{d}
	 * \\end{align*} }
	 * 
	 * @param origin  The ray origin in world coordinates.
	 * @param dir     The normalized(!) ray direction (corresponding to a specific pixel) in world coordinates [rd_x rd_y rd_z].
	 * @param cubmin  The cuboid's minimal planes given as [min_x, min_y, min_z] in world
	 *                coordinates.
	 * @param cubmax  The cuboid's maximal planes given as [max_x, max_y, max_z] in world
	 *                coordinates.
	 * @param distanceNearAndFar    Return values. In case of a hit: Positive distances (in world
	 *                coordinate units) of nearest and farthest plane intersection.
	 * @return        Boolean value which is true if the ray intersects with the bounding
	 *                box and false otherwise.
	 *
	 * @see <a href="http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm">http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm</a>
	 * @see <a href="http://dx.doi.org/10.1145/15922.15916">http://dx.doi.org/10.1145/15922.15916</a>
	 */
	public static boolean intersectRayWithCuboid(final SimpleVector origin, final SimpleVector dir, final SimpleVector cubmin, final SimpleVector cubmax, final double[] distanceNearAndFar) {
		// input checks
		assert (origin.getLen() == 3) : new IllegalArgumentException("origin has to be a 3-vector!");
		assert (dir.getLen() == 3) : new IllegalArgumentException("dir has to be a 3-vector!");
		assert (Math.abs(dir.normL2() - 1.0) < 10.0*CONRAD.DOUBLE_EPSILON) : new IllegalArgumentException("Direction vector must be of length 1:" + dir);
		assert (cubmin.getLen() == 3) : new IllegalArgumentException("cubmin has to be a 3-vector!");
		assert (cubmax.getLen() == 3) : new IllegalArgumentException("cubmax has to be a 3-vector!");

		// output checks
		assert (distanceNearAndFar.length == 2) : new IllegalArgumentException("tntf output must be a 2-array!");

		// init near and far intersection distance
		distanceNearAndFar[0] = Double.NEGATIVE_INFINITY;
		distanceNearAndFar[1] = Double.POSITIVE_INFINITY;

		// intersect with bounding box's 6 planes
		for (int i = 0; i < 3; ++i) { // loop over bb's x, y, and z plane pairs

			// check whether ray is parallel to the two i planes
			if (Math.abs(dir.getElement(i)) < CONRAD.DOUBLE_EPSILON) {

				// check whether ray is not between the two i planes, i.e., if it misses the box
				if (origin.getElement(i) < cubmin.getElement(i) || origin.getElement(i) > cubmax.getElement(i)) return false;

			} else {

				// compute the two intersections
				double t1 = (cubmin.getElement(i) - origin.getElement(i))/dir.getElement(i);
				double t2 = (cubmax.getElement(i) - origin.getElement(i))/dir.getElement(i);

				// sort t1 and t2 so that t1 is intersection with near plane, t2 with far plane (t1 <= t2)
				if (t1 > t2) {
					// swap the two values if necessary
					final double ttmp = t1;
					t1 = t2;
					t2 = ttmp;
				}

				// keep largest near intersection
				if (t1 > distanceNearAndFar[0]) distanceNearAndFar[0] = t1;

				// keep smallest far intersection
				if (t2 < distanceNearAndFar[1]) distanceNearAndFar[1] = t2;

				// check if box was missed
				if (distanceNearAndFar[0] > distanceNearAndFar[1] + CONRAD.FLOAT_EPSILON) return false;

				// check if box is behind the ray
				if (distanceNearAndFar[1] < 0) return false;
			}
		}

		// survived all tests, bounding box was hit by ray
		return true;
	}

	/**
	 * Compute the geometric center of a set of points
	 * @param list the set of points
	 * @return the geometric center
	 */
	public static PointND getGeometricCenter(ArrayList<PointND> list){
		int dim = list.get(0).getDimension();
		double [] temp = new double [list.get(0).getDimension()];
		for (int i = 0; i < list.size(); i++){
			for (int j = 0; j < dim; j++){
				temp[j] += list.get(i).get(j);
			}
		}
		for (int j = 0; j < dim; j++){
			temp[j] /= list.size();
		}
		return new PointND(temp);
	}

	/**
	 * Compute the geometric center of an iterator of points
	 * @param list the set of points
	 * @return the geometric center
	 */
	public static PointND getGeometricCenter(Iterator<PointND> list){
		PointND p = list.next();
		int count = 1;
		int dim = p.getDimension();
		double [] temp = new double [dim];
		for (int j = 0; j < dim; j++){
			temp[j] += p.get(j);
		}
		while (list.hasNext()){
			p = list.next();
			if (p!= null){
				count ++;
				for (int j = 0; j < dim; j++){
					temp[j] += p.get(j);
				}
			}
		}
		for (int j = 0; j < dim; j++){
			temp[j] /= count;
		}
		return new PointND(temp);
	}


	/**
	 * Extract points from an ImageProcessor which exceed a certain value
	 * 
	 * @param houghSpace the ImageProcessor
	 * @param offset the threshold for extraction
	 * @return the list of candidate points
	 */
	public static ArrayList<PointND> extractCandidatePoints(ImageProcessor houghSpace, double offset){
		ArrayList<PointND> candidate = new ArrayList<PointND>();
		for (int j = 0; j< houghSpace.getHeight(); j++){
			for (int i = 0; i< houghSpace.getWidth(); i++){
				if (houghSpace.getPixelValue(i, j) > offset) {
					PointND point = new PointND(i, j);
					candidate.add(point);
				}
			}
		}
		return candidate;
	}
	
	
	/**
	 * Threshold image and create binary mask
	 * 
	 * @param img the ImageProcessor
	 * @param offset the threshold
	 * @return the binary image
	 */
	public static ImageProcessor thresholdImage(ImageProcessor img, double offset){
		byte[] pixels = new byte[img.getWidth()*img.getHeight()];
		ImageProcessor imp = new BinaryProcessor(new ByteProcessor(img.getWidth(), img.getHeight(), pixels));
		for (int j = 0; j< img.getHeight(); j++){
			for (int i = 0; i< img.getWidth(); i++){
				if (img.getPixelValue(i, j) > offset) {
					imp.set(i,j,255);
				}
			}
		}
		return imp;
	}
	
	

	/**
	 * Extracts cluster centers from an ordered List of points. Points must be ordered first with respect to x, then to y coordinate. Algorithm assumes that only one point may appear in the same row, i.e.,  all clusters must be separable via the y direction.
	 * A cluster center is then computed as the geometric center of the points in the same cluster. Algorithm is fast, but very restricted.
	 * @param pointList the list of candidate points
	 * @param distance the minimal distance between clusters
	 * @return the list of cluster centers
	 */
	public static ArrayList<PointND> extractClusterCenter(ArrayList<PointND> pointList, double distance){
		return extractClusterCenter(pointList, distance, true);
	}
	
	/**
	 * Extracts cluster centers from an ordered List of points. 
	 * A cluster center is then computed as the geometric center of the points in the same cluster.
	 * @param pointList the list of candidate points
	 * @param distance the minimal distance between clusters
	 * @param breakOption option to break after distance exceeded once. Only applicable if points are ordered in the correct manner.
	 * @return the list of cluster centers
	 */
	public static ArrayList<PointND> extractClusterCenter(ArrayList<PointND> pointList, double distance, boolean breakOption){
		ArrayList<PointND> centerPoint = new ArrayList<PointND>();
		while (pointList.size() > 0){
			PointND reference = pointList.get(0);
			ArrayList<PointND> currentSubset = new ArrayList<PointND>();
			//currentSubset.add(reference);
			for (int i = pointList.size()-1; i >= 0; i--){
				PointND current = pointList.get(i);
				if (current.euclideanDistance(reference) < distance){
					currentSubset.add(current);
					pointList.remove(i);
				} else {
					// points are ordered first in x and then in y direction
					// hence, end of current cluster if more than distance away in y direction
					if (breakOption) if (Math.abs(reference.get(1) - current.get(1)) > distance) break;
				}
			}
			centerPoint.add(getGeometricCenter(currentSubset));
		}
		return centerPoint;
	}

	public static PointND getGeometricCenter(PointND[] pts) {
		int dim = pts[0].getDimension();
		double [] temp = new double [pts[0].getDimension()];
		for (int i = 0; i < pts.length; i++){
			for (int j = 0; j < dim; j++){
				temp[j] += pts[i].get(j);
			}
		}
		for (int j = 0; j < dim; j++){
			temp[j] /= pts.length;
		}
		return new PointND(temp);
	}
	
	/**
	 * Creates a triangle mesh for a planar set of points. The geometric center is estimated and each subsequent set of points is connected with the center to form the mesh.
	 * Note that the points must be neighboring points in the ArrayList. Such a set of points can be obtained using for example a convex hull algorithm. This method can also be used to close a BSplineSurface.
	 * @param points the points
	 * @return the list of triangles
	 * @see ConvexHull
	 * @see edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline
	 */
	public static ArrayList<Triangle> createTrianglesFromPlanarPointSet(ArrayList<PointND> points){
		return createTrianglesFromPlanarPointSet(points, "", null);
	}
	
	/**
	 * Creates a triangle mesh for a planar set of points. The geometric center is estimated and each subsequent set of points is connected with the center to form the mesh.
	 * Note that the points must be neighboring points in the ArrayList. Such a set of points can be obtained using for example a convex hull algorithm. This method can also be used to close a BSplineSurface.
	 * @param points the points
	 * @return the list of triangles
	 * @see ConvexHull
	 * @see edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline
	 */
	public static ArrayList<Triangle> createTrianglesFromPlanarPointSet(ArrayList<PointND> points, String nameTag, BufferedWriter bwpoint){
		ArrayList<Triangle> mesh = new ArrayList<Triangle>();
		if (points.size() > 0) {
			PointND center = General.getGeometricCenter(points);
			for (int i = 1; i < points.size(); i++){
				try{
					Triangle t = new Triangle(center, points.get(i), points.get(i-1));
					if (bwpoint != null){
						try {
							PointND pt = center;
							bwpoint.write("Center"+nameTag+"\t" + i+"id"+nameTag + "\t"+ +pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
							pt = points.get(i);
							bwpoint.write(nameTag+i+"\t"+i+"id"+nameTag + "\t" +pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
							pt = points.get(i-1);
							bwpoint.write(nameTag+(i-1)+"\t"+i+"id"+nameTag + "\t" +pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						} catch (IOException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
					t.setName(i+"id"+nameTag);
					mesh.add(t);
				} catch (Exception e){
					if (!e.getLocalizedMessage().contains("direction vector")){
						System.out.println(e.getLocalizedMessage());
					}
				}
			}
			try{
				Triangle t = new Triangle(center, points.get(0), points.get(points.size()-1));
				if (bwpoint != null){
					try {
						PointND pt = center;
						bwpoint.write("Center"+nameTag+"\t" + 0+"id"+nameTag + "\t"+ +pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						pt = points.get(0);
						bwpoint.write(nameTag+0+"\t"+0+"id"+nameTag + "\t" +pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						pt = points.get(points.size()-1);
						bwpoint.write(nameTag+(points.size()-1)+"\t"+0+"id"+nameTag + "\t" +pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				t.setName(0+"id"+nameTag);
				mesh.add(t);
			} catch (Exception e){
				if (!e.getLocalizedMessage().contains("direction vector")){
					System.out.println(e.getLocalizedMessage());
				}
			}
		}
		return mesh;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
