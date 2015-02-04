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
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.DoublePrecisionPointUtil;

public class MeshViewer extends OpenGLViewer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1481644355541714966L;
	private boolean enableCullling =false;
	ArrayList<Triangle> triangles = new ArrayList<Triangle>();
	ArrayList<Color> colors = null;

	public MeshViewer(String title, ArrayList<Triangle> triangles) {
		this(title, triangles, false);
	}
	
	public MeshViewer(String title, ArrayList<Triangle> triangles, boolean culling) {
		super(title);
		enableCullling = culling;
		this.setSize(640, 480);
		ArrayList<PointND> edgePoints = new ArrayList<PointND>();
		for (Triangle t : triangles) {
			edgePoints.add(t.getA());
			edgePoints.add(t.getB());
			edgePoints.add(t.getC());
		}
		PointND center = DoublePrecisionPointUtil
				.getGeometricCenter(edgePoints);
		Translation translation = new Translation(center.getAbstractVector()
				.negated());
		double max = 0;
		for (PointND p : edgePoints) {
			p.applyTransform(translation);
			for (int i = 0; i < 3; i++) {
				if (Math.abs(p.get(i)) > max) {
					max = Math.abs(p.get(i));
				}
			}
		}
		AffineTransform affineTransform = new AffineTransform(
				SimpleMatrix.I_3.dividedBy(max / 1.0),
				new SimpleVector(0, 0, 0));
		for (Triangle t : triangles) {
			t.applyTransform(translation);
			t.applyTransform(affineTransform);
		}
		this.triangles = triangles;
	}

	public void display(GLAutoDrawable arg0) {
		if (!initialized) {
			return;
		}
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
		// float modelView[] = new float[16];
		// gl.glMatrixMode(GL.GL_MODELVIEW);

		// gl.glPushMatrix();

		gl.glMatrixMode(GL4bc.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glFrustum(-1.5, 1.5, -1.5, 1.5, 5, 15);

		gl.glMatrixMode(GL4bc.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glTranslated(0, 0, -10);
		gl.glTranslatef(-translationX, -translationY, -translationZ);
		gl.glRotatef(-rotationX, 1.0f, 0.0f, 0.0f);
		gl.glRotatef(-(rotationY), 0.0f, 1.0f, 0.0f);

		// Internal Coordinates we want to use for visualization: (0,0,0) to
		// (1,1,1);
		for (int i = 0; i < triangles.size(); i++) {
			Triangle p = triangles.get(i);
			if (colors == null) {
				drawTriangle(gl, p, new Color(0, 1, 0));
			} else {
				Color col = colors.get(i);
				drawTriangle(gl, p, col);
			}

		}
		// drawCube(gl, new PointND(0,0,0), 0.01, 1, 1, 1);
	}

	public void dispose(GLAutoDrawable arg0) {
	}

	public void init(GLAutoDrawable arg0) {
		// Perform the default GL initialization
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		gl.glShadeModel(GL4bc.GL_SMOOTH);
		gl.glEnable(GL.GL_DEPTH_TEST);
		if (enableCullling) gl.glEnable(GL.GL_CULL_FACE);
		if (initialized) {
			return;
		}
		initialized = true;
	}

	/**
	 * @return the points
	 */
	public ArrayList<Triangle> getPoints() {
		return triangles;
	}

	/**
	 * @param points
	 *            the points to set
	 */
	public void setPoints(ArrayList<Triangle> points) {
		this.triangles = points;
	}

	/**
	 * @return the colors
	 */
	public ArrayList<Color> getColors() {
		return colors;
	}

	/**
	 * @param colors
	 *            the colors to set
	 */
	public void setColors(ArrayList<Color> colors) {
		this.colors = colors;
	}

	public static void main(String[] args) {
		ArrayList<Triangle> list = new ArrayList<Triangle>();
		double scale = 1000;

		// Triangle null: [185.5; 281.0; 81.0] [186.0; 281.5; 81.0] [185.5;
		// 282.0; 81.0]
		// Triangle null: [186.5; 281.0; 81.0] [187.0; 281.5; 81.0] [186.5;
		// 282.0; 81.0]
		// Triangle null: [186.0; 281.5; 81.0] [186.5; 281.0; 82.0] [187.0;
		// 281.5; 82.0]

		PointND x1 = new PointND(185.5, 281.0, 81.0);
		PointND y1 = new PointND(186.0, 281.5, 81.0);
		PointND z1 = new PointND(185.5, 282.0, 81.0);

		PointND x2 = new PointND(186.0, 281.5, 81.0);
		PointND y2 = new PointND(185.5, 282.0, 81.0);
		PointND z2 = new PointND(187.0, 281.5, 82.0);

		PointND x3 = new PointND(186.0, 281.5, 81.0);
		PointND y3 = new PointND(186.5, 281.0, 82.0);
		PointND z3 = new PointND(187.0, 281.5, 82.0);

		Triangle tri1 = new Triangle(x1, y1, z1);
		Triangle tri2 = new Triangle(x2, y2, z2);
		Triangle tri3 = new Triangle(x3, y3, z3);

		list.add(tri1);
		list.add(tri2);
		list.add(tri3);

		MeshViewer mv = new MeshViewer("Two Triangles", list, false);
		mv.setVisible(true);
	}
}
