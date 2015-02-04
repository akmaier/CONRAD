/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.mesh;

import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Contains methods operating on Mesh.class objects.
 * @author Mathias Unberath
 *
 */
public abstract class MeshUtil {
	
	/**
	 * Shifts the centroid of a mesh to the coordinate origin.
	 * @param m The input mesh.
	 * @return The centered mesh.
	 */
	public static Mesh centerToOrigin(Mesh m){
		Mesh c = new Mesh();
		c.setConnectivity(m.getConnectivity());
		c.setPoints(m.getPoints());
		
		SimpleMatrix pts = m.getPoints();
		// calculate mean
		SimpleVector mean = new SimpleVector(c.dimension);
		for(int i = 0; i < c.numPoints; i++){
			mean.add(pts.getRow(i));
		}
		mean.divideBy(c.numPoints);
		// subtract mean
		for(int i = 0; i < c.numPoints; i++){
			pts.getRow(i).subtract(mean);
		}
		c.setPoints(pts);
		
		return c;
	}
	
	/**
	 * Calculates the centroid of the vertices.
	 * @param m The mesh.
	 * @return The centroid.
	 */
	public static SimpleVector getCenter(Mesh m){		
		SimpleMatrix pts = m.getPoints();
		// calculate mean
		SimpleVector mean = new SimpleVector(m.dimension);
		for(int i = 0; i < m.numPoints; i++){
			mean.add(pts.getRow(i));
		}
		mean.divideBy(m.numPoints);
		
		return mean;
	}
	
	/**
	 * Scales the mesh in a way such that the sum of all coordinates is 1.
	 * @param m Input mesh.
	 * @return The scaled mesh.
	 */
	public static Mesh normalize(Mesh m){
		SimpleMatrix pts = m.getPoints();
		double s = 0;
		
		for(int i = 0; i < pts.getRows(); i++){
			for(int j = 0; j < pts.getCols(); j++){
				s += Math.pow(pts.getElement(i, j), 2);
			}
		}
		s = Math.sqrt(s);
		for(int i = 0; i < pts.getRows(); i++){
			for(int j = 0; j < pts.getCols(); j++){
				pts.multiplyElementBy(i, j, 1 / s);
			}
		}
		Mesh r = new Mesh();
		r.setConnectivity(m.getConnectivity());
		r.setPoints(pts);
		return r;
	}
	
	/**
	 * Finds the rotation needed to map the Mesh in m to the Mesh in f in a least squares sense. 
	 * @param f The reference.
	 * @param m To be rotated.
	 * @param shift Determines whether or not the mesh should be shifted to zero origin before rotation.
	 * @return A rotated version of m.
	 */
	public static Mesh rotate(Mesh f, Mesh m, boolean shift){
		Mesh fNorm = normalize(f);
		SimpleMatrix m1 = fNorm.getPoints();
		Mesh mNorm = normalize(m);
		SimpleMatrix m2 = mNorm.getPoints();
		
		// create matrix containing information about both point-clouds m1^T * m2
		SimpleMatrix m1Tm2 = SimpleOperators.multiplyMatrixProd(m1.transposed(), m2);
		// perform SVD such that:
		// m1^T * m2 = U sigma V^T
		DecompositionSVD svd = new DecompositionSVD(m1Tm2, true, true, true);
		// exchange sigma with new matrix s having only +/- 1 as singular values
		// this allows only for rotations but no scaling, e.g. sheer
		// signum is the same as in sigma, hence reflections are still taken into account
		int nColsS = svd.getS().getCols();
		SimpleMatrix s = new SimpleMatrix(nColsS,nColsS);
		for(int i = 0; i < nColsS; i++){
			s.setElementValue(i, i, Math.signum(svd.getSingularValues()[i]));
		}
		// calculate rotation matrix such that:
		// H = V s U^T
		SimpleMatrix h = SimpleOperators.multiplyMatrixProd(svd.getV(), SimpleOperators.multiplyMatrixProd(s, svd.getU().transposed()));
		
		return rotate(m,h,shift);		
	}
	
	/**
	 * Finds the rotation needed to map the Mesh in m to the vertices in f in a least squares sense. 
	 * @param f The reference.
	 * @param m To be rotated.
	 * @param shift Determines whether or not the mesh should be shifted to zero origin before rotation.
	 * @return A rotated version of m.
	 */
	public static Mesh rotate(SimpleMatrix f, Mesh m, boolean shift){
		Mesh fNorm = new Mesh();
		fNorm.setConnectivity(m.getConnectivity());
		fNorm.setPoints(f);
		fNorm = centerToOrigin(fNorm);
		fNorm = normalize(fNorm);
		SimpleMatrix m1 = fNorm.getPoints();
		
		Mesh mNorm = centerToOrigin(m);
		mNorm = normalize(m);
		SimpleMatrix m2 = mNorm.getPoints();
		
		// create matrix containing information about both point-clouds m1^T * m2
		SimpleMatrix m1Tm2 = SimpleOperators.multiplyMatrixProd(m1.transposed(), m2);
		// perform SVD such that:
		// m1^T * m2 = U sigma V^T
		DecompositionSVD svd = new DecompositionSVD(m1Tm2, true, true, true);
		// exchange sigma with new matrix s having only +/- 1 as singular values
		// this allows only for rotations but no scaling, e.g. sheer
		// signum is the same as in sigma, hence reflections are still taken into account
		int nColsS = svd.getS().getCols();
		SimpleMatrix s = new SimpleMatrix(nColsS,nColsS);
		for(int i = 0; i < nColsS; i++){
			s.setElementValue(i, i, Math.signum(svd.getSingularValues()[i]));
		}
		// calculate rotation matrix such that:
		// H = V s U^T
		SimpleMatrix h = SimpleOperators.multiplyMatrixProd(svd.getV(), SimpleOperators.multiplyMatrixProd(s, svd.getU().transposed()));
		
		return rotate(m,h,shift);		
	}
	
	/**
	 * Applies the rotation described in R to the vertices of the mesh.
	 * @param m The mesh to be rotated.
	 * @param r The rotation matrix.
	 * @param shift Determines whether or not the mesh should be shifted to zero origin before rotation.
	 * @return The rotated mesh.
	 */
	public static Mesh rotate(Mesh m, SimpleMatrix r, boolean shift){
		Mesh f = new Mesh();
		if(shift){
		SimpleVector mean = getCenter(m);
		Mesh shifted = shift(m, mean);
		
		SimpleMatrix pts = SimpleOperators.multiplyMatrixProd(shifted.getPoints(), r);
		f.setConnectivity(m.getConnectivity());
		f.setPoints(pts);
		f = shift(f, mean.negated());
		}else{
			SimpleMatrix pts = SimpleOperators.multiplyMatrixProd(m.getPoints(), r);
			f.setConnectivity(m.getConnectivity());
			f.setPoints(pts);
		}
		return f;
	}
	
	/**
	 * Adds the vector in s to every vertex in m.
	 * @param m The mesh.
	 * @param s	The shift.
	 * @return The shifted mesh.
	 */
	public static Mesh shift(Mesh m, SimpleVector s){
		assert(m.getPoints().getCols() == s.getLen()) : new Exception("Dimensions must agree.");
		Mesh c = new Mesh();
		c.setConnectivity(m.getConnectivity());
		SimpleMatrix pts = m.getPoints();
		for(int i = 0; i < pts.getRows(); i++){
			for(int j = 0; j < pts.getCols(); j++){
				pts.addToElement(i, j, s.getElement(j));
			}
		}
		c.setPoints(pts);
		return c;
	}

}

