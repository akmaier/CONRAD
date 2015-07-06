package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Stack;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.ProjectPointToLineComparator;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PhysicalPoint;
import edu.stanford.rsl.conrad.physics.detector.MaterialPathLengthDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

public class WatertightRayTracer extends AbstractRayTracer {

	protected int debug = 1;
	
	public WatertightRayTracer() {
		this.comparator = new ProjectPointToLineComparator();
		if (debug > 0) {
			System.out.println("This is the WATERTIGHT RAY TRACER");
		}
	}

	
	// Copied from SimpleRayTracer
	protected ArrayList<PhysicalObject> computeMaterialIntersectionSegments(PhysicalPoint [] rayList){
		if (Configuration.getGlobalConfiguration().getDetector() instanceof MaterialPathLengthDetector) {
			return computeMaterialIntersectionSegmentsMatPathLen(rayList);
		}
		
		ArrayList<PhysicalObject> segments = new ArrayList<PhysicalObject>();

		Stack<PhysicalObject> materialStack = new Stack<PhysicalObject>();
		materialStack.push(rayList[0].getObject());

		// Iterate over hits
		for (int k=1; k < rayList.length; k++){
			PhysicalObject obj = new PhysicalObject();
			Edge edge = new Edge(rayList[k-1], rayList[k]);
			obj.setShape(edge);
			if(materialStack.isEmpty()) {
				obj.setMaterial(scene.getBackgroundMaterial());
				obj.setNameString("Background");
			} else {
				// Take top of stack, but do not remove the top element
				PhysicalObject current = materialStack.peek();
				obj.setMaterial(current.getMaterial());
				obj.setNameString(current.getNameString());
			}
			segments.add(obj);
			PhysicalObject nextObject = rayList[k].getObject();

			if (materialStack.contains(nextObject)){
				materialStack.remove(nextObject);
			} else {
				materialStack.push(nextObject);
			}
		}

		return segments;
	}
	
	
	/**
	 * Compute material segments for material path length detector
	 */
	protected ArrayList<PhysicalObject> computeMaterialIntersectionSegmentsMatPathLen(PhysicalPoint [] rayList){
		ArrayList<PhysicalObject> segments = new ArrayList<PhysicalObject>();
		
		// Assuming that the list of hits is sorted in ray direction
		// For the Material Path Length Detector, we need to form segments for each material
		
		// Sort by material
		HashMap<Material, List<PhysicalPoint>> hitMap = new HashMap<>();
		for (int i=0; i<rayList.length; i++) {
			PhysicalPoint p = rayList[i];
			if (!hitMap.containsKey(p.getObject().getMaterial())) {
				hitMap.put(p.getObject().getMaterial(), new LinkedList<>());
			}
			hitMap.get(p.getObject().getMaterial()).add(p);
		}

		for (Map.Entry<Material, List<PhysicalPoint>> hitsByMaterial : hitMap.entrySet()) {
			List<PhysicalPoint> hits = hitsByMaterial.getValue();

			// Iterate over hits
			for (int k=0; k < hits.size(); k = k + 2){
				PhysicalObject obj = new PhysicalObject();
				Edge edge = new Edge(hits.get(k), hits.get(k+1));
				obj.setShape(edge);
				obj.setMaterial(hitsByMaterial.getKey());
				segments.add(obj);
			}
		}

		return segments;
	}
	
	
	/**
	 * The algorithm assumes that direction's largest absolute value is stored in the z-component
	 * @param vec direction of ray
	 * @return array of size 3 containing the permutation of the direction's components such that the third dimension holds the largest absolute value
	 */
	private int[] getMaxDirection(SimpleVector vec) {
		assert(vec.getLen() == 3);
		int[] idx = new int[3];
		
		if (Math.abs(vec.getElement(0)) > Math.abs(vec.getElement(1))) {
			if (Math.abs(vec.getElement(0)) > Math.abs(vec.getElement(2))) {
				// x > y and x > z
				idx[2] = 0;
				idx[0] = 1;
				idx[1] = 2;
			} else {
				// z > x > y
				idx[2] = 2;
				idx[0] = 0;
				idx[1] = 1;
			}
		} else {
			if (Math.abs(vec.getElement(1)) > Math.abs(vec.getElement(2))) {
				// y > x and y > z
				idx[2] = 1;
				idx[0] = 2;
				idx[1] = 0;
			} else {
				// z > y > x
				idx[2] = 2;
				idx[0] = 0;
				idx[1] = 1;
			}
		}
		
		// Swap kx and ky dimension if max direction is negative in order to preserve winding direction of triangles
		if (vec.getElement(idx[2]) < 0.d) {
			int temp = idx[0];
			idx[0] = idx[1];
			idx[1] = temp;
		}
		
		return idx;
	}
	


	/**
	 * Implementation of watertight algorithm as proposed by Sven Woop, Carsten Benthin and Ingo Wald
	 * See http://jcgt.org/published/0002/01/05/paper.pdf and https://github.com/embree/embree/blob/v1.1_watertight/rtcore/triangle/triangle1i_intersector1_watertight.h
	 */
	@Override
	public ArrayList<PhysicalPoint> intersectWithScene(AbstractCurve ray) {
		
		StraightLine line = (StraightLine) ray;
		
		double[] dir = line.getDirection().copyAsDoubleArray();
		PointND origin = line.getPoint();
		// Calculate dimensions where the ray direction is maximal
		int[] kPerm = getMaxDirection(line.getDirection());
		
		// For simplicity, move permutation indices into primitive variables
		int kx = kPerm[0];
		int ky = kPerm[1];
		int kz = kPerm[2];
		
		// Calculate shear constants
		double Sz = 1.0d / dir[kz];
		double Sx = dir[kx] * Sz;
		double Sy = dir[ky] * Sz;
		
		ArrayList<PhysicalPoint> hits = new ArrayList<>();
		
		Queue<AbstractShape> queue;
		// Iterate over objects of scene
		for (PhysicalObject obj : scene) {
			// This algorithm can only process triangles, which are usually embedded in a compound shape
			queue = new LinkedList<>();
			queue.add(obj.getShape());
			while (!queue.isEmpty()) {
				AbstractShape shape = queue.poll();
				if (shape instanceof CompoundShape) {
					CompoundShape cs = (CompoundShape) shape;

					// Test for hits on bounding box. In case the ray does not even hit the bounding box, we will not find an intersection with a triangle
					if (cs.getHitsOnBoundingBox(ray).size() > 0) {
						queue.addAll(cs);
					}
				}
				else if (!(shape instanceof Triangle)) {
					System.err.println("Only triangles can be intersected by the watertight algorithm. As the current shape is not a triangle, it will be skipped.");
				}
				else {
					Triangle triangle = (Triangle) shape;
					
					// Calculate vertices relative to ray origin
					PointND A = new PointND(SimpleOperators.subtract(triangle.getA().getAbstractVector(), origin.getAbstractVector()));
					PointND B = new PointND(SimpleOperators.subtract(triangle.getB().getAbstractVector(), origin.getAbstractVector()));
					PointND C = new PointND(SimpleOperators.subtract(triangle.getC().getAbstractVector(), origin.getAbstractVector()));
					
					// Perform shear and scale of vertices
					double[] aCoords = new double[A.getDimension()-1];
					double[] bCoords = new double[B.getDimension()-1];
					double[] cCoords = new double[C.getDimension()-1];
					
					aCoords[0] = A.get(kx) - Sx * A.get(kz);
					aCoords[1] = A.get(ky) - Sy * A.get(kz);
					bCoords[0] = B.get(kx) - Sx * B.get(kz);
					bCoords[1] = B.get(ky) - Sy * B.get(kz);
					cCoords[0] = C.get(kx) - Sx * C.get(kz);
					cCoords[1] = C.get(ky) - Sy * C.get(kz);
					
					// Calculate scaled barycentric coordinates
					double U = cCoords[0] * bCoords[1] - cCoords[1] * bCoords[0];
					double V = aCoords[0] * cCoords[1] - aCoords[1] * cCoords[0];
					double W = bCoords[0] * aCoords[1] - bCoords[1] * aCoords[0];
					
					// Perform edge tests
					// All scaled barycentric coordinates need to have the same sign
					if ((U < 0.d || V < 0.d || W < 0.d) && (U > 0.d || V > 0.d || W > 0.d)) {
						continue;
					}
					
					// Calculate determinant
					double det = U + V + W;
					
					if (CONRAD.SMALL_VALUE > Math.abs(det)) {
						// det is almost zero
						// The ray seems to be co-planar to the triangle surface
						continue;
					}
					
					// Calculate scaled z-coordinates of vertices and use them to calculate the hit distance
					double Az = Sz * A.get(kz);
					double Bz = Sz * B.get(kz);
					double Cz = Sz * C.get(kz);
					double T = U * Az + V * Bz + W * Cz;
					
					// Handle negative determinant
					if (det >= 0 && T < 0.d) {
						continue;
					} else if (det < 0 && T >= 0.d) {
						continue;
					}
					
					// Normalize barycentric coordinates and hit distance
					U = U / det;
					V = V / det;
					W = W / det;
					T = T / det;
					
					// Hit has the coordinates [0, 0, T]
					// Inverting the transformation which was applied to all vertices in the beginning yields:
					double[] hitCoords = new double[3];
					hitCoords[0] = T * dir[0] + origin.get(0);
					hitCoords[1] = T * dir[1] + origin.get(1);
					hitCoords[2] = T * dir[2] + origin.get(2);
					
					PhysicalPoint hit = new PhysicalPoint(hitCoords);
					// Calculating the dot product of the triangle's normal and the ray direction can help to determine whether the ray entered or left the object at this hit point
					hit.setHitOrientation(SimpleOperators.multiplyInnerProd(triangle.getNormal(), line.getDirection()));
					hit.setObject(obj);
					hits.add(hit);
				}
			}
			// Cleanup for next physical object
			queue.clear();
		}
		
		if (hits == null || hits.size() == 0) {
			return null;
		}
		
		return hits;

	}

}
