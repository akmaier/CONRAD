/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.tutorial.physics;


import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.media.opengl.GL;
import javax.media.opengl.GL2GL3;
import javax.media.opengl.GL4bc;
import javax.media.opengl.GLAutoDrawable;

import edu.stanford.rsl.apps.gui.opengl.OpenGLViewer;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.rendering.AbstractScene;

public class XRayViewer extends OpenGLViewer {


	private static final long serialVersionUID = 8218145790254271538L;

	private ArrayList<PointND> points = null;
	private ArrayList<Edge> edges = null;
	private ArrayList<Color> colors = null;
	
	private double maxPoint = 0;

	private AbstractScene scene;

	private Box source;

	
	
	public XRayViewer(String title, String fileName){
		this(title,readCSVFile(fileName),null);
	}
	
	public XRayViewer(String title, String fileName, double maxPoint){
		this(title,readCSVFile(fileName),null, maxPoint);
	}

	public XRayViewer(String title, ArrayList<PointND> points, ArrayList<Edge> edges){
		this(title, points,  edges,0);
	}
	
	public XRayViewer(String title, ArrayList<PointND> points, ArrayList<Edge> edges, double divideBy){
		super(title);
		this.setSize(1024, 1024);
		this.maxPoint = divideBy;
		boolean seachForMaxPoint = false;
		if (maxPoint == 0){
			seachForMaxPoint = true;
		}
		
		if (points != null){
			this.points = new ArrayList<PointND>();
			for(PointND p: points){
				PointND newP = new PointND(p.getAbstractVector().clone());
				this.points.add(newP);
				
				if (seachForMaxPoint){
					for (int i=0;i<3;i++){
						if (Math.abs(newP.get(i)) > maxPoint){
							maxPoint = Math.abs(newP.get(i));
						}
					}
				}

			}
			for(PointND p: this.points){
				p.getAbstractVector().divideBy(maxPoint);
			}			
		}

		
		if (edges != null){
			this.edges = new ArrayList<Edge>();
			double max = 0;
			for(Edge e: edges){
				for (int i=0;i<3;i++){
					if (Math.abs(e.getPoint().get(i)) > max){
						max = Math.abs(e.getPoint().get(i));
					}
					
					if (Math.abs(e.getEnd().get(i)) > max){
						max = Math.abs(e.getEnd().get(i));
					}
				}
			}
			for(Edge e: edges){
				PointND start = e.getPoint();
				PointND end = e.getEnd();
				start.getAbstractVector().divideBy(max);
				end.getAbstractVector().divideBy(max);

				this.edges.add(new Edge(start,end));

			}
			
		}
	}
	
	
	public static ArrayList<PointND> readCSVFile(String filename){
		ArrayList<PointND> points = new ArrayList<PointND>();
		try {
			BufferedReader bf = new BufferedReader(new FileReader(filename));
			String line = bf.readLine();
			while(line != null) {
				line = bf.readLine();
				if (line!=null) {
					String[] coords = line.split(",");
					PointND point = new PointND(Double.parseDouble(coords[0]),Double.parseDouble(coords[1]),Double.parseDouble(coords[2]));
					point.getAbstractVector().dividedBy(1000.d);
					points.add(point);
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (NumberFormatException e){
			e.printStackTrace();
		}
		return points;
	}
	
	public double getMaxPoint(){
		return maxPoint;
	}

	public void display(GLAutoDrawable arg0) {
		if (!initialized)
		{
			return;
		}
		GL4bc gl = (GL4bc) arg0.getGL();
		gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);


		gl.glMatrixMode(GL4bc.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glFrustum( -1.5, 1.5, -1.5, 1.5, 5, 15 );
//		gl.glOrtho( -1.5, 1.5, -1.5, 1.5, 5, 15 );


		gl.glMatrixMode(GL4bc.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glTranslated(0, 0, -10);
		gl.glTranslatef(-translationX, -translationY, -translationZ);
		gl.glRotatef(-rotationX, 1.0f, 0.0f, 0.0f);
		gl.glRotatef(-(rotationY), 0.0f, 1.0f, 0.0f);

		//draw the scene layout

		drawScene(gl);
		
		//draw the beam source
		drawSource(gl);

		
		if (points != null){
			// Internal Coordinates we want to use for visualization: (0,0,0) to (1,1,1);
			for (int i=0; i< points.size(); i++){
				PointND p =  points.get(i);
				if  (colors == null){
					drawCube(gl, p, 0.001, 0, 1, 0);
				} else {
					Color col = colors.get(i);
					drawCube(gl, p, 0.001, col.getRed()/256.0, col.getGreen()/256.0, col.getBlue()/256.0);
				}

			}
		}
		
		if(edges != null) {
			for (int i=0; i< edges.size(); i++){
				drawLine(gl, edges.get(i));
			}
		}

		
		//draw coordinate axes
		int axisLength = 1;
		Edge xAxis = new Edge(new PointND(0,0,0), new PointND(axisLength,0,0));
		Edge yAxis = new Edge(new PointND(0,0,0), new PointND(0,axisLength,0));
		Edge zAxis = new Edge(new PointND(0,0,0), new PointND(0,0,axisLength));

		drawLine(gl, xAxis,new Color(1, 0.f, 0.f));
		drawLine(gl, yAxis,new Color(0.0f, 1.f, 0.0f));
		drawLine(gl, zAxis,new Color(0.0f, 0.0f, 1.0f));


	}

	private void drawScene(GL4bc gl) {
		if (this.scene != null){
			for (PhysicalObject obj : this.scene){
				if (obj.getShape() instanceof AbstractSurface){
					AbstractSurface b = (AbstractSurface)obj.getShape();
					Color c;
					if (obj.getNameString() != null && obj.getNameString().equals("detector")){
						c = new Color(1,0,0);
					} else {
						c = new Color(1,1,1);
					}
					
					gl.glPolygonMode( GL.GL_FRONT_AND_BACK, GL2GL3.GL_LINE );
					gl.glPushMatrix();
					gl.glScaled(1/getMaxPoint(), 1/getMaxPoint(), 1/getMaxPoint());
					CompoundShape cs = (CompoundShape) b.tessellate(1);
					if (cs != null)
						for (AbstractShape s: cs){
							drawTriangle(gl, (Triangle)s, c);
						}
					gl.glPopMatrix();	
					gl.glPolygonMode( GL.GL_FRONT_AND_BACK, GL2GL3.GL_FILL );
				}
			}
		}
	}
	
	private void drawSource(GL4bc gl){
		if (this.source == null) return;
		
		Color c = new Color(0,0,1);
		
		gl.glPushMatrix();
		gl.glScaled(1/getMaxPoint(), 1/getMaxPoint(), 1/getMaxPoint());
		CompoundShape cs = (CompoundShape) source.tessellate(1);
		if (cs != null)
			for (AbstractShape s: cs){
				drawTriangle(gl, (Triangle)s, c);
			}
		gl.glPopMatrix();	
	}
	

	
	public void drawCuboid(GL4bc gl, PointND location, double scalex, double scaley,double scalez,double colorR, double colorG, double colorB){
		gl.glPolygonMode( GL.GL_FRONT_AND_BACK, GL2GL3.GL_LINE );

		gl.glPushMatrix();
		gl.glTranslated(location.get(0)/getMaxPoint(), location.get(1)/getMaxPoint(), location.get(2)/getMaxPoint());
		gl.glScaled(scalex/getMaxPoint()/3, scaley/getMaxPoint()/3, scalez/getMaxPoint()/3);

		OpenGLViewer.makeCube(gl, colorR, colorG, colorB);
		gl.glPopMatrix();		
		gl.glPolygonMode( GL.GL_FRONT_AND_BACK, GL2GL3.GL_FILL );

	}
	
	public static void drawLine(GL4bc gl, Edge edge, Color color){
		gl.glColor3f(color.getRed(), color.getGreen(), color.getBlue());
		
		gl.glBegin(GL4bc.GL_LINE_STRIP);
		
		PointND point = edge.getPoint();
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		point = edge.getEnd(); 
		gl.glVertex3f((float)point.get(0), (float)point.get(1), (float)point.get(2));
		gl.glEnd();
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


	public void setScene(AbstractScene scene){
		this.scene = scene;
	}
	
	public void setSource(Box b) {
		this.source = b;
	}
}
