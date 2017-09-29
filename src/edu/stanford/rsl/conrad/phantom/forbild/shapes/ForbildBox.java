package edu.stanford.rsl.conrad.phantom.forbild.shapes;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.mathexpressions.Evaluator;

/**
 * <p>This class creates a surface from a <a href = "http://www.imp.uni-erlangen.de/forbild/english/forbild/index.htm">forbild</a>definition.</p>
 * <p>The expression [Box: x=originX;  y=originY;  z=originZ;  dx=length;  dy=width;  dz= height] defines a box located at (originX,originY,originZ).</p>
 * 
 * @author Rotimi .X. Ojo
 */
public class ForbildBox extends Box {

	private static final long serialVersionUID = 4063615482635066034L;
	 private PointND surfaceOrigin = new PointND(0,0,0);
	 private ArrayList<Plane3D> boundingPlanes =  new ArrayList<Plane3D>();
	 private double dx;
	 private double dy;
	 private double dz;
	

	public ForbildBox(String expression){
		 parseExpression(expression);
		 SimpleMatrix rot = new SimpleMatrix(3,3);
		 rot.identity();
		 transform = new AffineTransform(rot, new SimpleVector(3));
		 correctAndAddBoundingConditions();
		 surfaceOrigin.set(0, surfaceOrigin.get(0)-dx/2);
		 surfaceOrigin.set(1, surfaceOrigin.get(1)-dy/2);
		 surfaceOrigin.set(2, surfaceOrigin.get(2)-dz/2);
		 super.init(surfaceOrigin, dx, dy , dz);
	 }
	
	public ForbildBox(ForbildBox fb){
		super(fb);
		dx = fb.dx;
		dy = fb.dy;
		dz = fb.dz;
		surfaceOrigin = (fb.surfaceOrigin != null) ? fb.surfaceOrigin.clone() : null;
		if (fb.boundingPlanes != null){
			boundingPlanes = new ArrayList<Plane3D>();
			Iterator<Plane3D> it = fb.boundingPlanes.iterator();
			while (it.hasNext()) {
				Plane3D pl = it.next();
				boundingPlanes.add((pl!=null) ? new Plane3D(pl) : null);
			}
		}
		else{
			boundingPlanes = null;
		}
	}

	private void parseExpression(String expression) {
		expression = expression.trim();
		if(expression.charAt(0)=='(' && expression.charAt(expression.length()-1)==')'){
			expression = expression.substring(1,expression.length()-1);
		}	
		
		String props = expression.substring(expression.indexOf(':')+ 1).trim();
		
		Scanner sc = new Scanner(props);
		sc.useDelimiter(";");		
		while(sc.hasNext()){
			String currProp = sc.next().trim();
			if(currProp.charAt(0)== 'x'&& currProp.contains("=")){
				surfaceOrigin.set(0,Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1)));
			}else if(currProp.charAt(0)== 'y'&& currProp.contains("=")){
				surfaceOrigin.set(1,Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1)));
			}else if(currProp.charAt(0)== 'z' && currProp.contains("=")){
				surfaceOrigin.set(2,Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1)));
			}else if(currProp.indexOf("dx")==0){
				dx = Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1));
			}else if(currProp.indexOf("dy")==0){
				dy = Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1));
			}else if(currProp.indexOf("dz")==0){
				dz = Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1));
			}else if(currProp.contains("r") && (currProp.contains(">") || currProp.contains("<"))){				
				boundingPlanes.add(Evaluator.getPlane(currProp));
			}
		}
	}
	
	/**
	 * Moves bounding planes from world space to the space of the bounded object
	 * Creates a bounding condition using this new plane and updates the superclass
	 */
	private void correctAndAddBoundingConditions() {
		Iterator<Plane3D> it = boundingPlanes.iterator();
		while(it.hasNext()){
			Plane3D currPlane = it.next();
			currPlane.applyTransform(transform.inverse());
			addBoundingCondition(new HalfSpaceBoundingCondition(currPlane));
		}		
	}
	
	@Override
	public AbstractShape clone() {
		return new ForbildBox(this);
	}
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/