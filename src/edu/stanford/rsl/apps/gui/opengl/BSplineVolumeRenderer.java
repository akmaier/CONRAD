/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.apps.gui.opengl;


import java.awt.Color;
import java.io.IOException;
import java.util.ArrayList;

import javax.media.opengl.GL;
import javax.media.opengl.GL4bc;
import javax.media.opengl.GLAutoDrawable;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.DoublePrecisionPointUtil;

/**
 * Class to render Surface B-Splines in an OpenGL viewer.
 * @author Mathias Unberath
 *
 */
public class BSplineVolumeRenderer extends OpenGLViewer{

	ArrayList<SurfaceBSpline> splines;
	ArrayList<Triangle> triangles;
	ArrayList<Color> colors;
	double sampling = 30;
	
	private static final long serialVersionUID = 2752954430998793647L;

	public BSplineVolumeRenderer(String filename) throws IOException {
		super(filename);
		this.setSize(640, 480);
		splines = SurfaceBSpline.readSplinesFromFile(filename);
		setUp(sampling, sampling);
	}
	
	public BSplineVolumeRenderer(SurfaceBSpline... spline){
		super("Spline rendering.");
		this.setSize(640, 480);
		splines = new ArrayList<SurfaceBSpline>();
		for(int i = 0; i < spline.length; i++){
			splines.add(spline[i]);
		}
		setUp(sampling, sampling);
	}
	
	public BSplineVolumeRenderer(double samplingU, double samplingV, SurfaceBSpline... spline){
		super("Spline rendering.");
		this.setSize(640, 480);
		splines = new ArrayList<SurfaceBSpline>();
		for(int i = 0; i < spline.length; i++){
			splines.add(spline[i]);
		}
		setUp(samplingU, samplingV);
	}
	
	private void setUp(double samplingU, double samplingV){
		this.triangles = new ArrayList<Triangle>();	
		
		for( int i = 0; i < splines.size(); i++){
			CompoundShape cs = (CompoundShape) splines.get(i).tessellateMesh(samplingU, samplingV, 2);
			for(int j = 0; j < cs.size(); j++){
				try{
					CompoundShape cs2 = (CompoundShape) cs.get(j);
					for(int k = 0; k < cs2.size(); k++){
						triangles.add((Triangle)cs2.get(k));
					}
				}catch(Exception e){
					//
				}					
			}
		}
		modify();
	}
	
	private void modify(){
		ArrayList<PointND> centerPoints = new ArrayList<PointND>();
		for (Triangle t : triangles) {
			centerPoints.add(t.getPoint());
		}
		PointND center = DoublePrecisionPointUtil.getGeometricCenter(centerPoints);
		Translation translation = new Translation(center.getAbstractVector().negated());
		double max = 0;
		for (Triangle t : triangles) {
			t.applyTransform(translation);
			PointND p = t.getPoint();
			for (int i = 0; i < 3; i++) {
				if (Math.abs(p.get(i)) > max) {
					max = Math.abs(p.get(i));
				}
			}
		}
		AffineTransform affineTransform = new AffineTransform(
				SimpleMatrix.I_3.dividedBy(max / 1.0),new SimpleVector(0, 0, 0));
		for (Triangle t : triangles) {
			// t.applyTransform(translation);
			t.applyTransform(affineTransform);
		}
	}
	
	public void display(GLAutoDrawable arg0) {
		if (!initialized) {
			return;
		}
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
		
		gl.glMatrixMode(GL4bc.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glFrustum(-1.5, 1.5, -1.5, 1.5, 5, 15);

		gl.glMatrixMode(GL4bc.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glTranslated(0, 0, -10);
		gl.glTranslatef(-translationX, -translationY, -translationZ);
		gl.glRotatef(-rotationX, 1.0f, 0.0f, 0.0f);
		gl.glRotatef(-(rotationY), 0.0f, 1.0f, 0.0f);

		// Internal Coordinates we want to use for visualization: (0,0,0) to (1,1,1);
		for (int i = 0; i < triangles.size(); i++) {
			Triangle p = triangles.get(i);
			if (colors == null) {
					drawTriangle(gl, p, Color.WHITE);
			} else {
					Color col = colors.get(i);
					drawTriangle(gl, p, col);
			}
		}
	}

	public void init(GLAutoDrawable arg0) {
		// Perform the default GL initialization
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		gl.glShadeModel(GL4bc.GL_SMOOTH);
		gl.glEnable(GL.GL_DEPTH_TEST);
		if (initialized) {
			return;
		}
		initialized = true;		
	}

	public void update(SurfaceBSpline... spline){
		splines = new ArrayList<SurfaceBSpline>();
		for(int i = 0; i < spline.length; i++){
			splines.add(spline[i]);
		}
		setUp(sampling, sampling);
	}
	
	public void dispose(GLAutoDrawable arg0) {
		// TODO Auto-generated method stub
		
	}

}