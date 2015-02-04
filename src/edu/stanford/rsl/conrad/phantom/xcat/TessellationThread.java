package edu.stanford.rsl.conrad.phantom.xcat;

import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.parallel.ParallelThread;
import edu.stanford.rsl.conrad.utils.TessellationUtil;

/**
 * Thread to tessellate a SurfaceBSpline or TimeVariantSurfaceBSpline. 
 * 
 * @author akmaier
 *
 */
public class TessellationThread extends ParallelThread {

	private Object tessellationObject;
	private double time;
	private CompoundShape scene;

	/**
	 * @return the mesh
	 */
	public CompoundShape getObjects() {
		return scene;
	}

	public TessellationThread (Object tessellationObject, double time){
		this.tessellationObject = tessellationObject;
		this.time = time;
	}

	@SuppressWarnings("rawtypes")
	@Override
	public void execute(){
		Object tObject = this;
		scene = new CompoundShape();
		while (tObject != null){ 
			if (((Iterator)this.tessellationObject).hasNext()){
				tObject = ((Iterator)this.tessellationObject).next();
				int elementCountU = (int)TessellationUtil.getSamplingU((AbstractSurface) tObject);
				int elementCountV = (int)TessellationUtil.getSamplingV((AbstractSurface) tObject);
				if (tObject instanceof SurfaceBSpline){
					SurfaceBSpline spline = (SurfaceBSpline)tObject;
					scene.add(spline.tessellateMesh(elementCountU, elementCountV));
				}
				if (tObject instanceof TimeVariantSurfaceBSpline){
					TimeVariantSurfaceBSpline spline = (TimeVariantSurfaceBSpline)tObject;
					AbstractShape mesh = spline.tessellateMesh(elementCountU, elementCountV, time);
					scene.add(mesh);			
				}
			} else {
				tObject = null;
			}
		}
	}

	@Override
	public String getProcessName() {
		return "Tessellation Thread";
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/