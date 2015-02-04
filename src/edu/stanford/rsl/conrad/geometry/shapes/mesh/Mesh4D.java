/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.mesh;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.UniformCubicBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.motion.estimation.EstimateCubic2DSpline;

/**
 * This class represents a time varying mesh. The time variation is described as splines so that each time-point can be 
 * sampled in a smooth fashion.
 * @author Mathias Unberath
 *
 */
public class Mesh4D {

	/**
	 * The ArrayList containing the mesh's vertices corresponding to the phases.
	 */
	private ArrayList<SimpleMatrix> meshes;
	/**
	 * The ArrayList containing the time-resolving splines for each vertex.
	 */
	private ArrayList<UniformCubicBSpline> splines;
	/**
	 * The number of meshes which is assumed equal to the phases, i.e. sampled time-steps.
	 */
	private int numMeshes;
	/**
	 * Dimension of the vertices;
	 */
	private int dimension;
	/**
	 * The number of vertices in the mesh.
	 */
	private int numPoints;
	/**
	 * The number of connections in the connectivity information, e.g. triangles.
	 */
	private int numConnections;
	/**
	 * The matrix containing the connectivity information.
	 */
	private SimpleMatrix triangles;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	/** 
	 * Constructs the object and initializes the class members.
	 */
	public Mesh4D(){
		this.meshes = new ArrayList<SimpleMatrix>();
		this.splines = new ArrayList<UniformCubicBSpline>();
		this.numMeshes = 0;
	}
	
	/**
	 * Adds a mesh's vertices to the ArrayList and increments the counter.
	 * When the first mesh is added, all necessary class members are set.
	 * @param m The mesh to add.
	 */
	public void addMesh(Mesh m){
		if(this.numMeshes == 0){
			this.triangles = m.getConnectivity();
			
			this.numConnections = m.numConnections;
			this.dimension = m.dimension;
			this.numPoints = m.numPoints;
		}else{
			assert(this.numConnections == m.numConnections) : new IllegalArgumentException("Input mesh does not fit the meshes already present.");
			assert(this.dimension == m.dimension) : new IllegalArgumentException("Input mesh does not fit the meshes already present.");
			assert(this.numPoints == m.numPoints) : new IllegalArgumentException("Input mesh does not fit the meshes already present.");
		}
		
		meshes.add(m.getPoints());
		
		numMeshes++;
	}
	
	/**
	 * Calculates the spline curves describing the vertex motion and stores them in the class member.
	 */
	public void calculateSplines(){
		for(int i = 0; i < numPoints; i++){
			ArrayList<PointND> points = new ArrayList<PointND>();
			for(int j = 0; j < numMeshes; j++){
				points.add(new PointND(meshes.get(j).getRow(i).copyAsDoubleArray()));
				//System.out.println(meshes.get(j).getRow(i).getElement(0) + " "+meshes.get(j).getRow(i).getElement(1)+" "+meshes.get(j).getRow(i).getElement(2));
			}
			points.add(points.get(0));
			
			EstimateCubic2DSpline fs = new EstimateCubic2DSpline(points);
			UniformCubicBSpline bs = fs.estimateUniformCubic(points.size()-3-1);
			//eval(bs,points);
			this.splines.add(bs);
		}
	}
	
	/**
	 * Method to evaluate the quality of a spline fit.
	 * @param bs The spline to be evaluated
	 * @param points The points that decide the quality.
	 */
	@SuppressWarnings("unused")
	private void eval(UniformCubicBSpline bs, ArrayList<PointND> points){
		double err = 0;
		for(int i = 0; i < points.size(); i++){
			double kj = (float)i/(points.size()-1);
			SimpleVector res = new SimpleVector(bs.evaluate(kj).getAbstractVector());
			res.subtract(points.get(i).getAbstractVector());
			err += res.normL2();
			System.out.println(res.getElement(0) + " " + res.getElement(1) + " " + res.getElement(2) + " " +err);
		}
		VisualizationUtil.createSplinePlot(bs).show();
	}
		
	/**
	 * Samples the time-variant splines list at a certain time t = [0,1] and returns the corresponding mesh.
	 * @param time The time to be sampled.
	 * @return The mesh corresponding to the time.
	 */
	public Mesh evaluateSplines(double time){
		Mesh m = new Mesh();
		m.setConnectivity(this.triangles);
		
		SimpleMatrix pts = new SimpleMatrix(numPoints, dimension);
		for(int i = 0; i < numPoints; i++){
			pts.setRowValue(i, new SimpleVector(splines.get(i).evalFast(time)));
		}
		m.setPoints(pts);
		return m;
	}
	
	/**
	 * Samples the time variant meshes at a certain time t=[0,1] and returns the corresponding mesh.
	 * Sampling is done using linear interpolation.
	 * @param time
	 * @return
	 */
	public Mesh evaluateLinearInterpolation(double time){
		Mesh m = new Mesh();
		m.setConnectivity(this.triangles);
		
		int low = (int)Math.floor(time * numMeshes);
		int high;
		if(low == numMeshes-1){
			high = 0;
		}else{
			high = low + 1;
		}
		double key = time*numMeshes - low;
		
		SimpleMatrix pts = new SimpleMatrix(numPoints, dimension);
		for(int i = 0; i < numPoints; i++){
			pts.setRowValue(i, interp(meshes.get(low).getRow(i), meshes.get(high).getRow(i), key));
		}
		m.setPoints(pts);
		return m;
	}
	
	/**
	 * Linearly interpolates between two simplevectors.
	 * @param low Floor of interp.
	 * @param high Ceil of interp.
	 * @param key Value of interp.
	 * @return The interpolated SimpleVector.
	 */
	private SimpleVector interp(SimpleVector low, SimpleVector high, double key){
		SimpleVector res = new SimpleVector(low.getLen());
		for(int i = 0; i < low.getLen(); i++){
			double val = (1-key)*low.getElement(i) + key*high.getElement(i);
			res.setElementValue(i, val);
		}
		return res;
	}
	
	/**
	 * Getter for the connectivity.
	 * @return The connectivity.
	 */
	public SimpleMatrix getFaces(){
		return this.triangles;
	}
	
	
	
}
