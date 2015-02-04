package edu.stanford.rsl.conrad.phantom.forbild.shapes;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.numerics.mathexpressions.Evaluator;

/**
 * <p>This class creates a surface from a <a href = "http://www.imp.uni-erlangen.de/forbild/english/forbild/index.htm">forbild</a>definition.</p>
 * <p>The expression [Sphere: x=originX;  y=originY;  z=originZ;  r = radius; axis(a_x, a_y, a_z)] defines a uniform sphere with center of mass at (originX,originY,originZ),.</p>
 *
 * @author Rotimi .X. Ojo
 */
public class ForbildSphere extends Sphere{

	private static final long serialVersionUID = 4059584393163102548L;
	
	private double radius;
	private PointND surfaceOrigin = new PointND(0,0,0);
	private ArrayList<Plane3D> boundingPlanes =  new ArrayList<Plane3D>();;
	
	public ForbildSphere(String expression){		
		parseExpression(expression);
		super.init(radius, surfaceOrigin);
		Iterator<Plane3D> it = boundingPlanes.iterator();
		while(it.hasNext()){
			Plane3D currPlane = it.next();
			currPlane.applyTransform(transform.inverse());
			addBoundingCondition(new HalfSpaceBoundingCondition(currPlane));
		}
	}
	
	public ForbildSphere(ForbildSphere fs){
		super(fs);
		radius = fs.radius;
		surfaceOrigin = (fs.surfaceOrigin != null) ? fs.surfaceOrigin.clone() : null;
		if (fs.boundingPlanes != null){
			boundingPlanes = new ArrayList<Plane3D>();
			Iterator<Plane3D> it = fs.boundingPlanes.iterator();
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
			if(currProp.charAt(0)== 'x'){
				surfaceOrigin.set(0,Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1)));
			}else if(currProp.charAt(0)== 'y'){
				surfaceOrigin.set(1,Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1)));
			}else if(currProp.charAt(0)== 'z'){
				surfaceOrigin.set(2,Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1)));
			}else if(currProp.charAt(0)== 'r' && !(currProp.contains(">") || currProp.contains("<"))){
				this.radius = Evaluator.getValue(currProp.substring(currProp.indexOf('=')+1));
			}else if(currProp.contains("r") && (currProp.contains(">") || currProp.contains("<"))){
				boundingPlanes.add(Evaluator.getPlane(currProp));
			}
		}
	}

	@Override
	public AbstractShape clone() {
		return new ForbildSphere(this);
	}
	
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/