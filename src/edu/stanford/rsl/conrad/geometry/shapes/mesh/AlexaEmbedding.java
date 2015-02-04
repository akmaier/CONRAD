/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.mesh;

import java.util.Random;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class AlexaEmbedding {
	
	/**
	 * Dimension of the vertices.
	 */
	static final int dimension = 3;
	/**
	 * Triangular connectivity is needed for Alexa embedding.
	 */
	static final int connectivity = 3;
	/**
	 * Number of vertices in the mesh.
	 */
	private int nVertices;
	/**
	 * Number of faces in the mesh.
	 */	
	private int nFaces;
	/**
	 * Vertices in the mesh.
	 */
	private SimpleMatrix vertices;
	/**
	 * Faces in the mesh.
	 */
	private SimpleMatrix faces;
	/**
	 * The normals of the faces.
	 */
	private SimpleMatrix normals;
	/**
	 * The centers of the faces.
	 */
	private SimpleMatrix centers;
	/**
	 * Number of knots in parametric u direction.
	 */
	private int nU;
	/**
	 * Number of knots in parametric v direction.
	 */
	private int nV;
	
	
	
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	/**
	 * Construct object and set class member.
	 * @param v Vertices.
	 * @param f Faces.
	 * @param nKnotsU Number of knots in parametric u-direction.
	 * @param nKnotsV Number of knots in parametric v-direction.
	 */
	public AlexaEmbedding(SimpleMatrix v, SimpleMatrix f, int nKnotsU, int nKnotsV){
		assert(v.getCols() == dimension) : new Exception("Alexa embedding not defined for meshes with dimension other than 3.");
		assert(f.getCols() == connectivity) : new Exception("Alexa embedding not defined for non-triangular meshes.");
		assert(nKnotsU > 0 && nKnotsV > 0) : new Exception("Number of knots in u and v direction must be greater than 0");
		
		this.vertices = v;
		this.nVertices = v.getRows();
		this.faces = f;
		this.nFaces = f.getRows();
		
		this.nU = nKnotsU;
		this.nV = nKnotsV;
	}
	
	/**
	 * Calculates the normals and centers of mass for all triangle faces of the mesh.
	 */
	private void getNormalsAndCenters(){		
		this.centers = new SimpleMatrix(nFaces, dimension);
		this.normals = new SimpleMatrix(nFaces, dimension);
		
		SimpleVector a;
		SimpleVector b;
		
		for(int i = 0; i < nFaces; i++){
			// calculate two edges spanning the triangle and the normal of the triangle
			a = vertices.getRow((int)faces.getElement(i, 1));
			a.subtract(vertices.getRow((int)faces.getElement(i,0)));
			b = vertices.getRow((int)faces.getElement(i, 2));
			b.subtract(vertices.getRow((int)faces.getElement(i,0)));
			normals.setRowValue(i, crossProduct(a,b));
			// calculate the center of the triangle
			centers.setRowValue(i, vertices.getRow((int)faces.getElement(i,0)));
			centers.getRow(i).add(a.multipliedBy(1/connectivity),b.multipliedBy(1/connectivity));
		}
	}
	
	/**
	 * Calculates the cross product of the two SimpleVectors a and b, c = a x b.
	 * Note: only defined in 3 dimensions.
	 * @param a Left vector.
	 * @param b Right vector.
	 * @return The cross product.
	 */
	private SimpleVector crossProduct(SimpleVector a, SimpleVector b){
		assert(a.getLen() == 3 && a.getLen() == b.getLen()) : new Exception("Input not fulfilling the requirement dim(a) = dim(b) = 3.");
		SimpleVector c = new SimpleVector(a.getLen());
		
		c.setElementValue(0, + a.getElement(1)*b.getElement(2) - a.getElement(2)*b.getElement(1));
		c.setElementValue(1, - a.getElement(0)*b.getElement(2) + a.getElement(2)*b.getElement(0));
		c.setElementValue(2, + a.getElement(0)*b.getElement(1) - a.getElement(1)*b.getElement(0));
		
		return c;
	}
	
	/**
	 * Calculates the smallest enclosing sphere of the mesh using Jack Ritter's approximative algorithm described in:
	 * Ritter, Jack. "An efficient bounding sphere." Graphics gems. Academic Press Professional, Inc., 1990.
	 * @return The parameters of the enclosing sphere where the first n-1 elements of the array contain the center and the last element the radius.
	 */
	private double[] computeSmallestEnclosingSphere(){
		double[] radius = new double[dimension + 1];
		
		Random rand = new Random();
		int initIdx =  rand.nextInt(nVertices);
		int idx1 = initIdx;
		int idx2 = initIdx;
		
		// find point idx1 with largest distance from randomly drawn point
		double dist = 0;
		for(int i = 0; i < nVertices; i++){
			double d = getDistance(initIdx,i);
			if(d > dist){
				dist = d;
				idx1 = i;
			}
		}
		// find point with largest distance to idx1
		dist = 0;
		for(int i = 0; i < nVertices; i++){
			double d = getDistance(idx1,i);
			if(d > dist){
				dist = d;
				idx2 = i;
			}
		}
		// center of bounding sphere is middle of connecting line, radius is half of the distance
		SimpleVector center = getCenter(idx1, idx2);
		double r = getDistance(idx1, idx2) / 2;
		
		// check if all points are inside the sphere
		// if not, create new sphere containing both the old sphere and the point formerly outside		
		while(true){
			initIdx = checkIfPointsOutside(center,r);
			if(initIdx == -1){ // if all points inside break loop
				break;
			}else{ // calculate new center and radius
				SimpleVector diff = vertices.getRow(initIdx);
				diff.subtract(center);
				
				center.add(diff.multipliedBy((diff.normL2() - r)/2));
				r = getDistance(center,initIdx) / 2;
			}			
		}

		for(int i = 0; i < dimension; i++){
			radius[i] = center.getElement(i);
		}
		radius[dimension] = r;
		
		return radius;
	}
	
	/**
	 * Checks if any vertex is still outside of the sphere wither center v and radius r.
	 * @param v The sphere's center.
	 * @param r The sphere's radius.
	 * @return The index to the first point outside of the sphere or -1 if sphere is enclosing every point.
	 */
	private int checkIfPointsOutside(SimpleVector v, double r){
		double d = 0;
		int i = 0;
		while( d < r || i < nVertices ){
			d = getDistance(v, i);
			i++;
		}
		if(i == nVertices){
			return -1;
		}else{
			return (i - 1);
		}
	}
	
	/**
	 * Calculate the center of the line connecting two vertices stored in the SimpleMatrix as rows i and j.
	 * @param i Vertex index.
	 * @param j Vertex index.
	 * @return The center of the connecting line.
	 */
	private SimpleVector getCenter(int i, int j){
		SimpleVector v = new SimpleVector(dimension);
		SimpleVector w = new SimpleVector(dimension);
		v = vertices.getRow(i);
		v.subtract(vertices.getRow(j));
		w = vertices.getRow(i);
		w.add(v.dividedBy(2));
		return w;
	}
	
	/**
	 * Calculates the distance between a vertex stored in the SimpleMatrix as row j and the point in p.
	 * @param p Point.
	 * @param j Vertex index.
	 * @return The distance between the two vertices.
	 */
	private double getDistance(SimpleVector p, int j){
		p.subtract(vertices.getRow(j));
		return p.normL2();
	}
	
	/**
	 * Calculates the distance between two vertices stored in the SimpleMatrix as rows i and j.
	 * @param i Vertex index.
	 * @param j Vertex index.
	 * @return The distance between the two vertices.
	 */
	private double getDistance(int i, int j){
		SimpleVector v = new SimpleVector(dimension);
		v = vertices.getRow(i);
		v.subtract(vertices.getRow(j));
		return v.normL2();
	}

}
