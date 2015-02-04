/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.apps.gui.opengl;

import java.awt.Color;
import java.util.ArrayList;

import javax.media.opengl.GL;
import javax.media.opengl.GL4bc;
import javax.media.opengl.GLAutoDrawable;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.DoublePrecisionPointUtil;

public class PointCloudViewer extends OpenGLViewer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1481644355541714966L;
	ArrayList<PointND> points = new ArrayList<PointND>();
	ArrayList<Color> colors = null;


	public PointCloudViewer(String title, ArrayList<PointND> points){
		super(title);
		this.setSize(640, 480);
		PointND center = DoublePrecisionPointUtil.getGeometricCenter(points);
		double max = 0;
		for(PointND p: points){
			PointND newP = new PointND(p.getAbstractVector().clone());
			newP.getAbstractVector().subtract(center.getAbstractVector());
			this.points.add(newP);
			for (int i=0;i<3;i++){
				if (Math.abs(newP.get(i)) > max){
					max = Math.abs(newP.get(i));
				}
			}
		}
		for(PointND p: this.points){
			p.getAbstractVector().divideBy(max);
		}
	}

	public void display(GLAutoDrawable arg0) {
		if (!initialized)
		{
			return;
		}
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
		//float modelView[] = new float[16];
		//gl.glMatrixMode(GL.GL_MODELVIEW);



		//gl.glPushMatrix();

		gl.glMatrixMode(GL4bc.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glFrustum( -1.5, 1.5, -1.5, 1.5, 5, 15 );


		gl.glMatrixMode(GL4bc.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glTranslated(0, 0, -10);
		gl.glTranslatef(-translationX, -translationY, -translationZ);
		gl.glRotatef(-rotationX, 1.0f, 0.0f, 0.0f);
		gl.glRotatef(-(rotationY), 0.0f, 1.0f, 0.0f);


		// Internal Coordinates we want to use for visualization: (0,0,0) to (1,1,1);
		for (int i=0; i< points.size(); i++){
			PointND p =  points.get(i);
			if  (colors == null){
				drawCube(gl, p, 0.005, 0, 1, 0);
			} else {
				Color col = colors.get(i);
				drawCube(gl, p, 0.005, col.getRed()/256.0, col.getGreen()/256.0, col.getBlue()/256.0);
			}

		}
		displayAdditionalThings(panel);
	}
	
	public void displayAdditionalThings(GLAutoDrawable arg0){
		
	}

	public void dispose(GLAutoDrawable arg0) {
	}

	public void init(GLAutoDrawable arg0) {
		// Perform the default GL initialization
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		gl.glShadeModel (GL4bc.GL_SMOOTH);
		gl.glEnable(GL.GL_DEPTH_TEST);		
		if (initialized)
		{
			return;
		}
		initialized = true;
	}

	/**
	 * @return the points
	 */
	public ArrayList<PointND> getPoints() {
		return points;
	}

	/**
	 * @param points the points to set
	 */
	public void setPoints(ArrayList<PointND> points) {
		this.points = points;
	}

	/**
	 * @return the colors
	 */
	public ArrayList<Color> getColors() {
		return colors;
	}

	/**
	 * @param colors the colors to set
	 */
	public void setColors(ArrayList<Color> colors) {
		this.colors = colors;
	}

}
