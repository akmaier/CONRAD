/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.gui.opengl;

import java.awt.Frame;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.media.opengl.GL;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.media.opengl.fixedfunc.GLLightingFunc;
import javax.media.opengl.glu.GLU;
import javax.media.opengl.glu.GLUnurbs;
import javax.media.opengl.glu.gl2.GLUgl2;

import com.jogamp.opengl.util.FPSAnimator;
/**
 * @author Lloyd
 *
 * This class was written for the purpose of testing surface functionality in JOGL2
 * set the variable bDoNurbs to true and run the program, then set it to false and run.
 * I'm wandering if Nurbs rendering works in JOGL2.  It seems like its close but that
 * none of the lighting is working; by contrast use glEvalMesh2 method of rendering works
 *
 */
public class SurfaceTest implements GLEventListener {

	/* if this value is false it will draw surface using
	 * glEvalMesh2  
	 * if true then it will use
	 * NurbsRenderer
	 * The question is whether nurbs is properly implemented in JOGL2.
	 * on my machine the lighting functionality seems to work with EvalMesh but
	 * not with NurbsRenderer
	 */
	boolean bDoNurbs = true;

	GLU glu;
	Point3DF cameraUpDirection = new Point3DF(0.f, 1.f, 0.f);
	float xCamera = 0.f;
	float tval = 0.f;
	private float deltaT = .05f;
	Point3DF cameraLoc = new Point3DF(xCamera, 2.f, 8.f);
	Point3DF lookAtPt = new Point3DF(0.f, -1.f, 0.f);

	float red[] = { 0.8f, 0.1f, 0.0f, 1.0f };
	float green[] = { 0.0f, 0.8f, 0.2f, 1.0f };
	float blue[] = { 0.f, 0.1f, 0.8f, 0.5f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float[] ctlarray = new float[] {
			-1.5f,-1.5f,4f,
			-0.5f,-1.5f,2.f,
			0.5f,-1.5f,-1.f,
			1.5f,-1.5f,2.f,

			-1.5f,-0.5f,1f,
			-0.5f,-0.5f,3.f,
			0.5f,-0.5f,0.f,
			1.5f,-0.5f,-1f,

			-1.5f,0.5f,4f,
			-0.5f,0.5f,0f,
			0.5f,0.5f,3f,
			1.5f,0.5f,4f,

			-1.5f,1.5f,-2.f,
			-0.5f,1.5f,-2.0f,
			0.5f,1.5f,0.0f,
			1.5f,1.5f,-1.0f
	};
	// Point3DF class is only for readability purposes
	class Point3DF {
		float x;
		float y;
		float z;
		Point3DF(float a, float b, float c) {
			x=a;
			y=b;
			z=c;
		}
	}

	public void display(GLAutoDrawable drawable) {
		update(drawable.getGL().getGL2());
		render(drawable);
	}
	private void update(GL2 gl) {
		gl.glLoadIdentity();
		glu.gluLookAt(cameraLoc.x, cameraLoc.y, cameraLoc.z,
				lookAtPt.x, lookAtPt.y, lookAtPt.z,
				cameraUpDirection.x, cameraUpDirection.y, cameraUpDirection.z);
		tval += deltaT;
		cameraLoc.x = (float) Math.sin(tval);
	}
	private void render(GLAutoDrawable drawable) {
		if (bDoNurbs) {
			drawNurbSurface(drawable);
		}
		else {
			drawSurface(drawable);
		}
	}

	public void dispose(GLAutoDrawable drawable) {
		System.out.println("--dispose--");
	}

	public void init(GLAutoDrawable drawable) {
		System.out.println("--init--");
		GL2 gl = drawable.getGL().getGL2();
		glu = new GLU();

		System.out.println("INIT GL IS: " + gl.getClass().getName());
		if (!bDoNurbs) {
			gl.glMap2f(GL2.GL_MAP2_VERTEX_3, 0, 1, 3, 4, 0, 1, 12, 4, ctlarray, 0);
		}
		gl.glEnable(GL2.GL_MAP2_VERTEX_3);
		gl.glEnable(GL2.GL_AUTO_NORMAL);
		gl.glMapGrid2f(20, 0.0f, 1.0f, 20, 0.0f, 1.0f);

		setupLighting(drawable, gl);
		float fovy=40.f;
		float aspect=1.f;
		float znear=1.f;
		float zfar=20f;
		glu.gluPerspective(fovy, aspect, znear, zfar);

		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();
		glu.gluLookAt(cameraLoc.x, cameraLoc.y, cameraLoc.z,
				lookAtPt.x, lookAtPt.y, lookAtPt.z,
				cameraUpDirection.x, cameraUpDirection.y, cameraUpDirection.z);
	}
	private void setupLighting(GLAutoDrawable drawable, GL2 gl) {
		int paramsOffset = 0;
		float pos[] = { 2.0f, -2.0f, 10.0f, 0.0f };
		float ambient[] = {0.5f, 0.5f, 0.5f, 1.0f};
		float diffuse[] = {0.9f, 0.9f, 0.9f, 1.0f};
		float specular[] = {0.05f, 0.05f, 0.99f, 1.0f};

		gl.glLightfv(GL2.GL_LIGHT0, GL2.GL_POSITION, pos, 0);
		gl.glEnable(GL2.GL_LIGHTING);
		gl.glEnable(GL2.GL_LIGHT0);
		gl.glEnable(GL2.GL_DEPTH_TEST);
		gl.glEnable(GL2.GL_AUTO_NORMAL);
		gl.glEnable(GL2.GL_NORMALIZE);

		gl.glLightfv(GL2.GL_LIGHT0, GL2.GL_AMBIENT, ambient, paramsOffset);
		gl.glLightfv(GL2.GL_LIGHT0, GL2.GL_DIFFUSE, diffuse, paramsOffset);
		gl.glLightfv(GL2.GL_LIGHT0, GL2.GL_SPECULAR, specular, paramsOffset);
		gl.glLightfv(GL2.GL_LIGHT0, GL2.GL_POSITION, pos, paramsOffset);
		gl.glLightModeli(GL2.GL_LIGHT_MODEL_LOCAL_VIEWER, GL2.GL_TRUE);

		gl.glClearColor(1f, 1f, 1f, 1f);
		gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();
		gl.glViewport(0, 0, drawable.getSurfaceWidth(), drawable.getSurfaceHeight());
		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glLoadIdentity();
		if (gl instanceof GLLightingFunc) {
			((GLLightingFunc) gl).glMaterialfv(GL.GL_FRONT, GLLightingFunc.GL_AMBIENT_AND_DIFFUSE, blue, 0);
			((GLLightingFunc) gl).glMaterialfv(GL.GL_FRONT, GLLightingFunc.GL_SPECULAR, white, 0);
			float[] medHiShiny = new float[] {80.0f};
			((GLLightingFunc) gl).glMaterialfv(GL.GL_FRONT, GLLightingFunc.GL_SHININESS, medHiShiny, 0);
			((GLLightingFunc) gl).glShadeModel(GLLightingFunc.GL_FLAT);
		}
	}

	public void reshape(GLAutoDrawable drawable, int x, int y, int width,
			int height) {
		System.out.println("--reshape--");
	}

	private void drawSurface(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
		gl.glPushMatrix();
		gl.glEvalMesh2(GL2.GL_FILL, 0, 20, 0, 20);
		gl.glPopMatrix();
		gl.glFlush();
	}
	private void drawNurbSurface(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);

		GLUgl2 glug12 = new GLUgl2();
		GLUnurbs nurbsRenderer = glug12.gluNewNurbsRenderer();
		//glug12.gluNurbsProperty(nurbsRenderer, , arg2)
		glug12.gluBeginSurface(nurbsRenderer);
		int uknot_cnt=8;
		float[] uknot = new float[] {0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f};
		int vknot_cnt = 8;
		float[] vknot = uknot;
		int ustride=4*3;
		int vstride=3;
		int uorder=4;
		int vorder=4;
		int evl_type=GL2.GL_MAP2_NORMAL;
		//glug12.gluNurbsSurface(nurbsRenderer, uknot_cnt, uknot, vknot_cnt, vknot, ustride, vstride, ctlarray, uorder, vorder, evl_type);
		evl_type=GL2.GL_MAP2_VERTEX_3;
		glug12.gluNurbsSurface(nurbsRenderer, uknot_cnt, uknot, vknot_cnt, vknot, ustride, vstride, ctlarray, uorder, vorder, evl_type);
		glug12.gluEndSurface(nurbsRenderer);
	}
	/**
	 * @param args
	 */
	 public static void main(String[] args) {
		GLProfile.initSingleton();
		GLProfile glp = GLProfile.getDefault();
		GLCapabilities caps = new GLCapabilities(glp);
		GLCanvas canvas = new GLCanvas(caps);

		Frame frame = new Frame("Test Surface rendering in JOGL 2 using nurbs or eval-mesh");
		frame.setSize(300, 300);
		frame.add(canvas);
		frame.setVisible(true);

		// by default, an AWT Frame doesn't do anything when you click
		// the close button; this bit of code will terminate the program when
		// the window is asked to close
		frame.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});
		canvas.addGLEventListener(new SurfaceTest());
		FPSAnimator animator = new FPSAnimator(canvas, 5);
		//animator.add(canvas);
		animator.start();
	 }
} 
