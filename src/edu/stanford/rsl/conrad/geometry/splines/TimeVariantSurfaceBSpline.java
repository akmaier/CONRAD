/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.splines;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.OpenCLParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.motion.ParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.TessellationUtil;

public class TimeVariantSurfaceBSpline extends AbstractSurface {

	/**
	 * 
	 */
	private static int tesselationCount = 0;
	private static int tesselationCount1 = 0;
	protected ArrayList<SurfaceBSpline> timeVariantShapes;
	private String title;
	private static final long serialVersionUID = 6323388850451960550L;
	protected BSpline timeSpline;
	protected int tPoints;
	protected boolean clockwise;

	public boolean isClockwise(){
		return clockwise;
	}

	public TimeVariantSurfaceBSpline(ArrayList<SurfaceBSpline> splines){
		timeVariantShapes = splines;
		init();
	}

	public TimeVariantSurfaceBSpline(SurfaceBSpline timeInvariant, MotionField motion, int timePoints, boolean addInitial){
		timeVariantShapes = new ArrayList<SurfaceBSpline>();
		ArrayList<PointND> initialPoints = timeInvariant.getControlPoints();
		// add initial state.
		if (addInitial) {
			timeVariantShapes.add(new SurfaceUniformCubicBSpline(initialPoints, timeInvariant.getUKnots(), timeInvariant.getVKnots()));
		}
		// Time information comes from the time variant spline(s)
		tPoints = timePoints;
		ArrayList<ArrayList<PointND>> newPoints = new ArrayList<ArrayList<PointND>>();
		if (motion instanceof ParzenWindowMotionField) {
			for (int t = 1; t < tPoints; t++){
				PointND [] pointsArray = new PointND[initialPoints.size()];
				initialPoints.toArray(pointsArray);
				ArrayList<PointND> pts = null;
				if (OpenCLUtil.isOpenCLConfigured()){
					try {
						CLContext context = OpenCLUtil.getStaticContext();
						CLDevice device = context.getMaxFlopsDevice();
						OpenCLParzenWindowMotionField openCLMotion = new OpenCLParzenWindowMotionField((ParzenWindowMotionField)motion, context, device);
						pts = openCLMotion.getPositions(0, ((double)t)/(tPoints-1), pointsArray);
					} catch (IOException e) {
						// Program not found.
						e.printStackTrace();
					}
				} else {
					pts = motion.getPositions(0, ((double)t)/(tPoints-1), pointsArray);
				}
				newPoints.add(pts);
			}
		} else {
			double [] times = new double [tPoints-1];
			for (int t = 1; t < tPoints; t++){
				newPoints.add(new ArrayList<PointND>());
				times[t-1] = ((double)t)/(tPoints-1);
			}
			for (int i = 0; i < initialPoints.size(); i++){
				ArrayList<PointND> pts = motion.getPositions(initialPoints.get(i), 0, times);
				for (int t = 1; t < tPoints; t++){
					newPoints.get(t-1).add(pts.get(t-1));
				}	
			}
		}
		for (int t = 1; t < tPoints; t++) {
			timeVariantShapes.add(new SurfaceUniformCubicBSpline(newPoints.get(t-1), timeInvariant.getUKnots(), timeInvariant.getVKnots()));
		}
		// Set properties for timeVariantShapes
		for (int t = 0; t < timeVariantShapes.size(); t++) {
			timeVariantShapes.get(t).setTitle(timeInvariant.getTitle());
			timeVariantShapes.get(t).setClockwise(timeInvariant.isClockwise());
		}
		init();
	}
	
	public TimeVariantSurfaceBSpline(TimeVariantSurfaceBSpline tvs){
		super(tvs);
		title = (tvs.title!=null) ? new String(tvs.title) : null;
		tPoints = tvs.tPoints;
		clockwise = tvs.clockwise;
		
		if (tvs.timeVariantShapes != null){
			Iterator<SurfaceBSpline> it = tvs.timeVariantShapes.iterator();
			timeVariantShapes = new ArrayList<SurfaceBSpline>();
			while (it.hasNext()) {
				SurfaceBSpline spl = it.next();
				timeVariantShapes.add((spl!=null) ? (SurfaceBSpline) spl.clone() : null);
			}
		}
		else{
			timeVariantShapes = null;
		}
		
		timeSpline = (tvs.timeSpline!=null) ? (BSpline)tvs.timeSpline.clone() : null;
	}

	private void init(){
		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		for (AbstractShape spline: timeVariantShapes) {
			max.updateIfHigher(spline.getMax());
			min.updateIfLower(spline.getMin());
		}
		tPoints = timeVariantShapes.size();
		setTitle(timeVariantShapes.get(0).getTitle());
		createTimeSpline();
		clockwise = timeVariantShapes.get(0).clockwise; // assumes all time steps have same orientation
	}

	private void createTimeSpline(){
		SimpleVector knotVector = new SimpleVector(4+timeVariantShapes.size());
		ArrayList<PointND> controlPoints = new ArrayList<PointND>();
		int count = 0;
		for (count = 0; count < 4; count++){
			knotVector.setElementValue(count, 0);
			controlPoints.add(new PointND(0, 0, 0));
		}
		for (count = 4; count < timeVariantShapes.size(); count++){
			knotVector.setElementValue(count, ((count - 3.0) / (timeVariantShapes.size() - 3.0)));
			controlPoints.add(new PointND(1, 1, 1));
		}
		for (count = timeVariantShapes.size(); count < timeVariantShapes.size() + 4; count++){
			knotVector.setElementValue(count, 1.0);
		}
		timeSpline = new UniformCubicBSpline(controlPoints, knotVector);
	}
	
	public int getNumberOfTimePoints(){
		return tPoints;
	}

	public ArrayList<PointND> getControlPoints(int time){
		return timeVariantShapes.get(time).getControlPoints();
	}

	@Override
	public void applyTransform(Transform t) {
		for (AbstractShape spline: timeVariantShapes){
			spline.applyTransform(t);
		}
	}

	@Override
	public int getDimension() {
		return timeVariantShapes.get(0).getDimension();
	}

	@Override
	public PointND evaluate(PointND u) {
		return evaluate(u.get(0), u.get(1), u.get(2));
	}

	public PointND evaluate(double u, double v, double t){
		double [] p = new double [timeVariantShapes.get(0).getControlPoints().get(0).getDimension()];

		double internal = ((t * timeSpline.getKnots().length) -3.0);
		int numPts = timeSpline.getControlPoints().size();
		double step = internal- Math.floor(internal);

		//System.out.println(u + " " + internal);
		double [] weights = UniformCubicBSpline.getWeights(step);
		for (int i=0;i<4;i++){
			double [] loc = (internal+i < 0)? ((SurfaceBSpline)timeVariantShapes.get(0)).evaluate(u, v).getCoordinates(): (internal+i>=numPts)? ((SurfaceBSpline)timeVariantShapes.get(numPts-1)).evaluate(u, v).getCoordinates() : ((SurfaceBSpline)timeVariantShapes.get((int) (internal+i))).evaluate(u, v).getCoordinates();
			for (int j = 0; j < loc.length; j++){
				p[j] += (loc[j] * weights[i]);
				//p[j] = loc[j];
			}
		}
		return new PointND(p);
	}

	@Deprecated
	public PointND evaluateFull(double u, double v, double t){
		SimpleVector point = new SimpleVector(getDimension()); 
		for (int i = 0; i < tPoints; i++){
			//System.out.println("Upoints: " + uPoints + " " + i);
			double weight = timeSpline.getWeight(t, i);
			point.add(((SurfaceBSpline)timeVariantShapes.get(i)).evaluate(u, v).getAbstractVector().multipliedBy(weight));
		}
		PointND revan = new PointND(point);
		return revan;
	}

	public PointND[] getRasterPoints(double samplingU, double samplingV, double time){
		PointND [] pts = new PointND[((int)samplingU) * ((int)samplingV)];
		for(double i =0 ; i < samplingU; i++){
			for (double j = 0; j < samplingV; j++){
				PointND p = evaluate(i/samplingU, j/ samplingV, time);
				pts[(int)(i*samplingV+j)] = p;
			}
		}
		return pts;
	}

	public AbstractShape tessellateMesh(double samplingU, double samplingV, double time){
		PointND [] pts = getRasterPoints(samplingU, samplingV, time);
		boolean write = false;
		BufferedWriter bwpoint = null;
		String filename = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_WRITE_MESH);
		if (filename != null) {
			String substring = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_WRITE_MESH_SURFACE_SUBSTRING);
			if (substring != null){
				write = getTitle().contains(substring);
			} 
			if (write){
				String splineName = getTitle().replace('*', '-');
				try {
					//bwpoint = new BufferedWriter(new FileWriter(filename + "_" + splineName +  "_" + time + ".points.txt"));
					bwpoint = new BufferedWriter(new FileWriter(filename + "_" + splineName +  "_" + tesselationCount + ".points.txt"));
					tesselationCount++;
					bwpoint.write(splineName+"\n");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					throw new RuntimeException(e);
				}
			}
		}
		CompoundShape superShape = new CompoundShape();
		ArrayList<PointND> lastSlice = new ArrayList<PointND>();
		ArrayList<PointND> firstSlice = new ArrayList<PointND>();
		for(double i =1 ; i < samplingU; i++){
			CompoundShape mesh =  new CompoundShape();
			double lasti = i - 1;
			if (lasti < 0) lasti = samplingU - 1;
			for (double j = 0; j < samplingV; j++){
				double lastj = j - 1;
				if (lastj < 0) lastj = samplingV - 1;
				if (i == samplingU-1) lastSlice.add(pts[(int)(i*samplingV+j)]);
				if (i == 1) firstSlice.add(pts[(int)((i-1)*samplingV+j)]);
				try{
					Triangle t1 = new Triangle(pts[(int)(lasti*samplingV+lastj)] ,
							pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)]);
					t1.setName(i+"x"+j+"u");
					if (write){
						PointND pt = pts[(int)(lasti*samplingV+lastj)];
						bwpoint.write(lasti+"x"+lastj+"\t"+i+"x"+j+"u\t"+pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						pt = pts[(int)(i*samplingV+lastj)];
						bwpoint.write(i+"x"+lastj+"\t"+i+"x"+j+"u\t"+pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						pt = pts[(int)(lasti*samplingV+j)];
						bwpoint.write(lasti+"x"+j+"\t"+i+"x"+j+"u\t"+pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
					}
					mesh.add(t1);
				} catch (Exception e){
					if (!e.getLocalizedMessage().contains("direction vector")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
				try{

					Triangle t2 = new Triangle(pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)] ,
							pts[(int)(i*samplingV+j)]);
					t2.setName(i+"x"+j+"l");
					if (write){
						PointND pt = pts[(int)(i*samplingV+lastj)];
						bwpoint.write(i+"x"+lastj+"\t"+i+"x"+j+"l\t"+pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						pt = pts[(int)(lasti*samplingV+j)];
						bwpoint.write(lasti+"x"+j+"\t"+i+"x"+j+"l\t"+pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
						pt = pts[(int)(i*samplingV+j)];
						bwpoint.write(i+"x"+j+"\t"+i+"x"+j+"l\t"+pt.get(0)+"\t"+pt.get(1)+"\t"+pt.get(2)+"\n");
					}
					mesh.add(t2);
				} catch (Exception e){
					if (!e.getLocalizedMessage().contains("direction vector")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
			}
			superShape.add(mesh);
		}
		superShape.addAll(General.createTrianglesFromPlanarPointSet(lastSlice, "firstSlice", bwpoint));
		superShape.addAll(General.createTrianglesFromPlanarPointSet(firstSlice, "lastSlice", bwpoint));
		if (write){
			try {
				bwpoint.flush();
				bwpoint.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//System.out.println("Triangles: " + superShape + " " + superShape.getInternalDimension() + " " + superShape.size());
		superShape.setName(getTitle());
		if (write){
			try {
				String splineName = getTitle().replace('*', '-');
				//BufferedWriter bw = new BufferedWriter(new FileWriter(filename + "_" + splineName +  "_" + time + ".txt"));
				BufferedWriter bw = new BufferedWriter(new FileWriter(filename + "_" + splineName +  "_" + tesselationCount1 + ".txt"));
				tesselationCount1++;
				bw.write(splineName+"\n");
				for (int i = 0; i < superShape.getInternalDimension(); i++){
					AbstractShape subShape = superShape.get(i);
					if (subShape instanceof CompoundShape) {
						CompoundShape cmpShape = (CompoundShape) subShape;
						for (int j=0;j <cmpShape.size(); j++){
							Triangle tri = (Triangle) cmpShape.get(j);
							bw.write(tri.toString()+"\n");
						}
					} else {
						bw.write(subShape.toString()+"\n");
					}
				}
				bw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return superShape;
	}

	@Override
	public int getInternalDimension() {
		return 3;
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		// TODO Auto-generated method stub
		return null;
	}

	public PointND[] getRasterPoints(int number, double time){
		int root  = (int) Math.sqrt(number);
		PointND [] points = new PointND[root*root];
		for (double u =0; u < root; u++){
			for (double v = 0; v < root; v++){
				points[(int)((v*root)+u)] = evaluate(u/root, v/root, time);
			}
		}
		return points;
	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		throw new RuntimeException("Not yet implemented");
	}

	/**
	 * @param title the title to set
	 */
	public void setTitle(String title) {
		this.title = title;
	}

	/**
	 * @return the title
	 */
	public String getTitle() {
		return title;
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	/**
	 * Returns a binary representation of a time variant surface spline:
	 * <pre>
	 * type
	 * total size in float values
	 * # of surface splines
	 * ID
	 * timeSpline
	 * surfaceSplines
	 * </pre>
	 * @return the binary representation for use with OpenCL
	 */
	public float[] getBinaryRepresentation() {
		float type = BSpline.SPLINE3DTO3D;
		float tPoints = this.tPoints;
		float ID = BSpline.getID(title);
		float [] timeSpline = this.timeSpline.getBinaryRepresentation();
		float [][] surfaceSplines = new float [this.tPoints][];
		int timeVariantsLength = 0;
		for (int i = 0; i < this.tPoints; i++){
			surfaceSplines[i] = this.timeVariantShapes.get(i).getBinaryRepresentation();
			timeVariantsLength += surfaceSplines[i].length;
		}
		int totalsize = 4 + timeSpline.length + timeVariantsLength;
		float [] binary = new float [totalsize];
		binary [0] = type;
		binary [1] = totalsize;
		binary [2] = ID;
		binary [3] = tPoints;
		int index = 4;
		for (int i = 0; i < timeSpline.length; i++){
			binary[index] = timeSpline[i];
			index++;
		}
		for (int i=0; i<surfaceSplines.length; i++){
			for (int j = 0; j< surfaceSplines[i].length; j++){
				binary[index] = surfaceSplines[i][j];
				index ++;
			}
		}
		return binary;
	}

	public ArrayList<SurfaceBSpline> getSplines() {
		return timeVariantShapes;
	}

	@Override
	public PointND evaluate(double u, double v) {
		return evaluate(u, v, 0);
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		return tessellateMesh(TessellationUtil.getSamplingU(this), TessellationUtil.getSamplingV(this), 0);
	}

	@Override
	public AbstractShape clone() {
		return new TimeVariantSurfaceBSpline(this);
	}
}
