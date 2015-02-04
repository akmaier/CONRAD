/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.gui.opengl;


import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.media.opengl.GL;
import javax.media.opengl.GL4bc;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.awt.GLJPanel;
import javax.swing.JFrame;

import com.jogamp.opengl.util.Animator;

import edu.stanford.rsl.conrad.cuda.MouseControlable;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;



public abstract class OpenGLViewer extends JFrame implements GLEventListener, MouseControlable {
	private static final long serialVersionUID = 5230114485975361359L;
	protected float rotationX;
	protected float rotationY;
	protected float translationX;
	protected float translationY;
	protected float translationZ =0;
	protected boolean initialized;
	private int width = 300;
	private int height = 300;
	private static Animator animator;
	protected GLJPanel panel;

	public OpenGLViewer (String title){
		super(title);
		panel = new GLJPanel();
		panel.addGLEventListener(this);
		panel.setPreferredSize(new Dimension(width, height));
		edu.stanford.rsl.conrad.cuda.MouseControl mouseControl = new edu.stanford.rsl.conrad.cuda.MouseControl(this);
		panel.addMouseMotionListener(mouseControl);
		panel.addMouseWheelListener(mouseControl);
		this.add(panel);
		pack();
		setVisible(true);
		boolean animate = true;
		if (animate) {
			animator = new Animator(panel);
			animator.setRunAsFastAsPossible(true);
			animator.start();
		}

		addWindowListener(new WindowAdapter()
		{
			public void windowClosing(WindowEvent e)
			{
				runExit();
			}
		});
	}

	/**
	 * Implementation of GLEventListener: Called then the GLAutoDrawable was
	 * reshaped
	 */
	public void reshape(
			GLAutoDrawable drawable, int x, int y, int width, int height)
	{
		this.width = width;
		this.height = height;

	}

	/**
	 * Draws a Cube at the position location
	 * @param gl the gldrawable
	 * @param location the loaction
	 * @param scale the scale of the cube
	 */
	public void drawCube(GL4bc gl, PointND location, double scale, double colorR, double colorG, double colorB){
		gl.glPushMatrix();
		gl.glTranslated(location.get(0), location.get(1), location.get(2));
		gl.glScaled(scale, scale, scale);

		OpenGLViewer.makeCube(gl, colorR, colorG, colorB);
		gl.glPopMatrix();		
	}
	
	public static void drawTriangleShape(GL4bc gl, AbstractShape shape){
		if (shape instanceof Triangle) {
			OpenGLViewer.drawTriangle(gl, (Triangle) shape);
		}
		if (shape instanceof CompoundShape) {
			CompoundShape cs = (CompoundShape) shape;
			for (AbstractShape s: cs){
				drawTriangleShape(gl, s);
			}
		}
	}

	public static void drawTriangle(GL4bc gl, Triangle triangle){
		Color color = new Color(0.9f, 0.5f, 0.2f);
		drawTriangle(gl, triangle, color);
	}
	
	public static void drawTriangle(GL4bc gl, Triangle triangle, Color color){
		PointND point = triangle.getA(); 
		gl.glColor3f(color.getRed(), color.getGreen(), color.getBlue());
		
		
		gl.glCullFace(GL.GL_BACK);
		gl.glBegin(GL.GL_TRIANGLE_FAN);
		point = triangle.getA(); 
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		point = triangle.getB(); 
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		point = triangle.getC(); 
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		gl.glEnd();
	}
	
	public static void drawLine(GL4bc gl, Edge edge){
		Color color = new Color(0.9f, 0.5f, 0.2f);
		gl.glColor3f(color.getRed(), color.getGreen(), color.getBlue());
		
//		gl.glCullFace(GL.GL_FRONT_AND_BACK);
		gl.glBegin(GL4bc.GL_LINE_STRIP);
		
		PointND point = edge.getPoint();
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		point = edge.getEnd(); 
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		gl.glEnd();
	}

	/**
	 * Stops the animator and calls System.exit() in a new Thread.
	 * (System.exit() may not be called synchronously inside one 
	 * of the JOGL callbacks)
	 */
	protected void runExit()
	{
		new Thread(new Runnable()
		{
			public void run()
			{
				animator.stop();
				System.exit(0);
			}
		}).start();
	}

	public void updateRotationX(double increment) {
		rotationX += increment;
	}

	public void updateRotationY(double increment) {
		rotationY += increment;
	}

	public void updateTranslationX(double increment) {
		translationX += increment;
	}

	public void updateTranslationY(double increment) {
		translationY += increment;
	}

	public void updateTranslationZ(double increment) {
		translationZ += increment;
	}

	public static void makeCube(GL4bc gl, double colorR, double colorG, double colorB){
		//gl.glNewList(2, GL.GL_COMPILE);
		gl.glColor3d(colorR, colorG, colorB);
		// Draw the sides of the cube
		gl.glCullFace(GL.GL_FRONT_AND_BACK);
		gl.glBegin(GL4bc.GL_QUAD_STRIP);
		gl.glVertex3d(3, 3, -3);
		gl.glVertex3d(3, -3, -3);
		gl.glVertex3d(-3, 3, -3);
		gl.glVertex3d(-3, -3, -3);
		gl.glVertex3d(-3, 3, 3);
		gl.glVertex3d(-3, -3, 3);
		gl.glVertex3d(3, 3, 3);
		gl.glVertex3d(3, -3, 3);
		gl.glVertex3d(3, 3, -3);
		gl.glVertex3d(3, -3, -3);
		gl.glEnd();
		// Draw the top and bottom of the cube
		gl.glBegin(GL4bc.GL_QUADS);
		gl.glVertex3d(-3, -3, -3);
		gl.glVertex3d(3, -3, -3);
		gl.glVertex3d(3, -3, 3);
		gl.glVertex3d(-3, -3, 3);
		gl.glVertex3d(-3, 3, -3);
		gl.glVertex3d(3, 3, -3);
		gl.glVertex3d(3, 3, 3);
		gl.glVertex3d(-3, 3, 3);
		gl.glEnd();
		//gl.glEndList();
	}

}
