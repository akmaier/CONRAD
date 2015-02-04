package edu.stanford.rsl.conrad.geometry.splines;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.LinearOctree;
import edu.stanford.rsl.conrad.geometry.shapes.compound.NestedOctree;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.OptimizableFunction;

/**
 * Class to model a surface made of BSplines.
 * 
 * 
 * @author akmaier
 *
 */
public class SurfaceBSpline extends AbstractSurface  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4942618494585588413L;
	protected SimpleVector uKnots;
	protected SimpleVector vKnots;
	protected int numberOfUPoints;
	protected int numberOfVPoints;
	protected ArrayList<PointND> points;
	protected BSpline [] vSplines;
	protected BSpline uSpline;
	protected int dimension;
	protected boolean clockwise;
	public static final int TESSELATE_COMPOUND_SHAPE = 0x1;
	public static final int TESSELATE_COMPOUND_OF_COMPOUND_SHAPES = 0x2;
	public static final int TESSELATE_COMPOUND_OF_OCTREES = 0x3;
	public static final int TESSELATE_LINEAR_OCTREE = 0x4;
	public static final int TESSELATE_NESTED_OCTREE = 0x5;
	private String title;

	public ArrayList<PointND> getControlPoints(){
		return points;
	}
	
	public SimpleVector getUKnots(){
		return uKnots;
	}
	
	public SimpleVector getVKnots(){
		return vKnots;
	}
	
	public void setUKnot(SimpleVector uknots){
		uKnots = uknots;
	}
	
	public void setVKnot(SimpleVector vknots){
		vKnots = vknots;
	}
	
	public boolean isClockwise(){
		return clockwise;
	}
	public void setClockwise(boolean in_clockwise){
		clockwise = in_clockwise;
	}
	
	public BSpline[] getVSplines(){
		return vSplines;
	}

	/**
	 * @return the title
	 */
	public String getTitle() {
		return title;
	}

	/**
	 * @param title the title to set
	 */
	public void setTitle(String title) {
		this.title = title;
	}

	protected synchronized void init(){
		dimension = points.get(0).getDimension();
		int degreeU = 0;
		while(uKnots.getElement(degreeU) == 0) degreeU++;
		degreeU --;
		int degreeV = 0;
		while(uKnots.getElement(degreeV) == 0) degreeV++;
		degreeV --;
		numberOfUPoints = uKnots.getLen() - ((degreeU+1));
		numberOfVPoints = vKnots.getLen() - ((degreeV+1));
		assert(numberOfUPoints * numberOfVPoints == points.size());
		ArrayList<PointND> temp = new ArrayList<PointND>();
		for(int i = 0; i < numberOfUPoints; i++){
			temp.add(points.get(0));
		}
		uSpline = new BSpline(temp, uKnots);
		vSplines = new BSpline[numberOfUPoints];
		for(int i = 0; i < numberOfUPoints; i++){
			temp = new ArrayList<PointND>();
			for(int j = 0; j < numberOfVPoints; j++){
				temp.add(points.get((i*numberOfVPoints)+j));
			}
			vSplines[i] = new BSpline(temp, vKnots);
		}
		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		for (PointND p: points) {
			max.updateIfHigher(p);
			min.updateIfLower(p);
		}
		generateBoundingPlanes();
	}
	
	protected boolean checkPointsClockwise(PointND a, PointND b, PointND center)
	{
		/*
		 * after: http://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
		 */
	    if (a.get(0) >= 0 && b.get(0) < 0)
	        return true;
	    if (a.get(0) == 0 && b.get(0) == 0)
	        return a.get(1) > b.get(1);

	    // compute the cross product of vectors (center -> a) x (center -> b)
	    double det = ((a.get(0)-center.get(0)) * (b.get(1)-center.get(1)) - (b.get(0) - center.get(0)) * (a.get(1) - center.get(1)));
	    if (det < 0) {
	        return true;
	    } else if (det > 0) {
	        return false;
	    }

	    // points a and b are on the same line from the center
	    // check which point is closer to the center
	    double d1 = (a.get(0)-center.get(0)) * (a.get(0)-center.get(0)) + (a.get(1)-center.get(1)) * (a.get(1)-center.get(1));
	    double d2 = (b.get(0)-center.get(0)) * (b.get(0)-center.get(0)) + (b.get(1)-center.get(1)) * (b.get(1)-center.get(1));
	    return d1 > d2;
	}
	
	/**
	 * Constructor for a surface BSpline. 
	 * 
	 * @param controlPoints the control points
	 * @param uKnots the knot vector in u direction
	 * @param vKnots the knot vector in v direction
	 */
	public SurfaceBSpline(ArrayList<PointND> controlPoints, SimpleVector uKnots, SimpleVector vKnots) {
		points = controlPoints;
		this.uKnots = uKnots;
		this.vKnots = vKnots;
		init();
		
/*		// Determination of spline orientation (increasing u,v clockwise or counterclockwise)
		// assumption: splines are slice by slice
		PointND center = new PointND(0,0,0);
		for (PointND p: points) {
			for (int d=0; d < 3; d++) center.set(d, p.get(d) / (double) points.size() + center.get(d));
		}
		int clockwisePoints=0;
		for (int i=0; i < points.size()-1;i++) {
			// actually incorrect calculation, checkPointsClockwise() only compares x,y coordinates
			// works for heart splines (by chance?)
			if (checkPointsClockwise(points.get(i), points.get(i+1), center))
				clockwisePoints++;
			else 
				clockwisePoints--;
		}
		clockwise = clockwisePoints > 0; // TODO: does not work for artery trees, nonconvex shapes
*/		
		// Determination of spline orientation (increasing u,v clockwise or counterclockwise)
		// assumption: splines are slice by slice with increasing z
		int clockwisePoints=0;
		for (int i=0; i < numberOfUPoints; i++) {
			PointND center = new PointND(0,0,0);
			int point_index = 0;
			for (int j=0; j < numberOfVPoints; j++) {
				point_index = i*numberOfVPoints+j;
				for (int d=0; d < 3; d++) center.set(d, points.get(point_index).get(d) / (double) numberOfVPoints + center.get(d));
			}
			for (int j=0; j < numberOfVPoints-1; j++) {
				point_index = i*numberOfVPoints+j;
				if (checkPointsClockwise(points.get(point_index), points.get(point_index+1), center))
					clockwisePoints++;
				else 
					clockwisePoints--;
			}
		}
		clockwise = clockwisePoints > 0;
		//System.out.println("Clockwisepoints " +clockwisePoints+ " for a total of "+points.size());
	}

	/**
	 * Constructor for a surface BSpline. 
	 * 
	 * @param list the control points
	 * @param uKnots the knot vector in u direction
	 * @param vKnots the knot vector in v direction
	 */
	public SurfaceBSpline(ArrayList<PointND> list, double[] uKnots,
			double[] vKnots) {
		this(list, new PointND(uKnots).getAbstractVector(), new PointND(vKnots).getAbstractVector());
	}

	/**
	 * Constructor for a surface BSpline. 
	 * 
	 * @param title the title of the BSpline
	 * @param list the control points
	 * @param uKnots the knot vector in u direction
	 * @param vKnots the knot vector in v direction
	 */
	public SurfaceBSpline(String title, ArrayList<PointND> list, double[] uKnots,
			double[] vKnots) {
		this(list, new PointND(uKnots).getAbstractVector(), new PointND(vKnots).getAbstractVector());
		this.title = title;
	}
	
	/**
	 * Constructor for a surface BSpline. 
	 * 
	 * @param title the title of the BSpline
	 * @param list the control points
	 * @param uKnots the knot vector in u direction
	 * @param vKnots the knot vector in v direction
	 */
	public SurfaceBSpline(String title, ArrayList<PointND> list, SimpleVector uKnots,
			SimpleVector vKnots) {
		this(list, uKnots, vKnots);
		this.title = title;
	}

	public static ArrayList<PointND> cloneList(List<PointND> list) {
		ArrayList<PointND> clone = new ArrayList<PointND>(list.size());
	    for(PointND item: list) clone.add(item.clone());
	    return clone;
	}
	
	public SurfaceBSpline(SurfaceBSpline spline) {
		this(cloneList(spline.points), new SimpleVector(spline.uKnots), new SimpleVector(spline.vKnots));
		min = (spline.min!=null) ? spline.min.clone() : null;
		max = (spline.max!=null) ? spline.max.clone() : null;
		title = spline.title;
	}

	@Override
	public PointND evaluate(double u, double v){
		SimpleVector point = new SimpleVector(dimension); 
		for (int i = 0; i < numberOfUPoints; i++){
			//System.out.println("Upoints: " + uPoints + " " + i);

			double weight = uSpline.getWeight(u, i);
			point.add(vSplines[i].evaluate(v).getAbstractVector().multipliedBy(weight));
		}
		PointND revan = new PointND(point);
		return revan;
	}

	@Override
	public int getDimension() {
		return dimension;
	}

	/**
	 * Reads a BSpline from a BufferedReader and returns it.
	 * @param reader the reader
	 * @return the surface BSpline
	 * @throws IOException
	 */
	public static SurfaceBSpline readBSpline(BufferedReader reader) throws IOException{
		boolean done = false;
		int uPoints = -1;
		int vPoints = -1;
		double [] vKnots = null;
		double [] uKnots = null;
		String last = "No title";
		String line = "No title";
		ArrayList<PointND> list = new ArrayList<PointND>();
		while (!done){
			if (!line.equals("")) last = line ;
			line = reader.readLine();
			if (line == null){
				throw new IOException("End of File reached");
			}
			if (line.contains(":M")){
				String [] elements = line.split("\\s+");
				uPoints = Integer.parseInt(elements[0]);
				//System.out.println("M = " + uPoints);
				break;
			}
		}
		while (!done){
			line = reader.readLine();
			if (line.contains(":M")){
				String [] elements = line.split("\\s+");
				uPoints = Integer.parseInt(elements[0]);
				//System.out.println("M = " + uPoints);
			}
			if (line.contains(":N")){
				String [] elements = line.split("\\s+");
				vPoints = Integer.parseInt(elements[0]);
				//System.out.println("N = " + vPoints);
			}
			if (line.contains("U Knot Vector")){
				vKnots = readKnotVector(reader, vPoints);
			}
			if (line.contains("V Knot Vector")){
				uKnots = readKnotVector(reader, uPoints);
			}
			if (line.contains("Control Points")){
				for (int i = 0; i < vPoints; i++){
					list.add(new PointND(read3DPoint(reader)));
				}
				for (int j = 1; j < uPoints; j++) {
					reader.readLine();
					for (int i = 0; i < vPoints; i++){
						list.add(new PointND(read3DPoint(reader)));
					}
				}
				//System.out.println("Last Point " + list.get(list.size()-1).toString());
				done = true;
				break;
			}
		}
		//System.out.println("SurfaceBSpline read: " + last);
		//System.out.println("#CP = " + list.size());
		//
		//return new SurfaceBSpline(last, list, uKnots, vKnots);
		if ((uKnots[0] == uKnots[1]) && (uKnots[2] == uKnots[3])) return new SurfaceUniformCubicBSpline(last, list, uKnots, vKnots);
		else return new SurfaceBSpline(last, list, uKnots, vKnots);
	}
	
	public static ArrayList<SurfaceBSpline> readSplinesFromFile(String filename) throws IOException{
		FileReader file = new FileReader(filename);
		BufferedReader bf = new BufferedReader(file);
		ArrayList<SurfaceBSpline> list = new ArrayList<SurfaceBSpline>();
		// read all surface splines in the file.
		boolean reading = true;
		while (reading) {
			try {
				SurfaceBSpline spline = SurfaceBSpline.readBSpline(bf);
				list.add(spline);
			} catch (IOException e){
				reading = false;
			}
		}
		return list;
	}

	private static double [] readKnotVector(BufferedReader reader, int knotPoints) throws IOException{
		double [] uKnots = null;
		int degree = 0;
		double number = 0;
		while (number == 0){
			number = readDouble(reader);
			degree++;
		}
		degree -= 2;
		//System.out.println("Degree: " + degree);
		uKnots = new double [degree + knotPoints +1];
		for (int i = 0; i < degree + 1; i++){
			uKnots[i] = 0;
			//System.out.println(uKnots[i]);
		}
		uKnots[degree + 1] = number;
		//System.out.println(uKnots[degree+1]);
		for (int i = degree +2; i< knotPoints +1 +degree; i++){
			uKnots[i] = readDouble(reader);
			//System.out.println(uKnots[i]);
		}
		return uKnots;
	}

	private static double readDouble(BufferedReader reader) throws IOException{
		String read = reader.readLine();
		String [] elements = read.split("\\s+");
		if (elements.length == 2) {
			return Double.parseDouble(elements[1]);
		} else {
			return Double.parseDouble(elements[0]);
		}
	}

	private static double [] read3DPoint(BufferedReader reader) throws IOException {
		double [] point  = new double [3];
		String read = reader.readLine();
		String [] elements = read.split("\\s+");
		if (elements.length == 4) {
			point[0] =  Double.parseDouble(elements[1]);
			point[1] =  Double.parseDouble(elements[2]);
			point[2] =  Double.parseDouble(elements[3]);
		} else {
			point[0] =  Double.parseDouble(elements[0]);
			point[1] =  Double.parseDouble(elements[1]);
			point[2] =  Double.parseDouble(elements[2]);
		}
		return point;
	}


	/**
	 * Computes approximate u, v coordinates to of the closest control point to the given point.
	 * @param point the point
	 * @return the approximate (u, v) coordinates of the control point.
	 */
	public double [] computeInitialUV(PointND point){
		int closeI = 0;
		int closeJ = 0;
		double minDistance = Double.MAX_VALUE;
		for (int i = 0; i < numberOfUPoints; i++){
			for (int j = 0; j < numberOfVPoints; j++){
				double distance = vSplines[i].getControlPoint(j).euclideanDistance(point);
				if (distance < minDistance){
					minDistance = distance;
					closeI = i;
					closeJ = j;
				}
			}

		}
		int offsetU = (uSpline.getDegree() + 1) /2;
		int offsetV = (vSplines[0].getDegree() + 1) /2;
		return new double[] {Math.log(uSpline.getKnotVectorEntry(closeI+offsetU)), Math.log(vSplines[closeI].getKnotVectorEntry(closeJ + offsetV))};
	}

	@Override
	public synchronized ArrayList<PointND> intersect(AbstractCurve other) {
		if (other instanceof StraightLine){
			StraightLine line = (StraightLine) other;
			// These two calls are not thread safe. Hence, synchronization is required.
			ArrayList<PointND> list = getHitsOnBoundingBox(line);

			//System.out.println(list.size());
			if (list.size() > 1) {
				boolean simple = false;
				ArrayList<PointND> intersectionPoints = new ArrayList<PointND>();
				if (!simple) {

					Edge connection = new Edge(list.get(0), list.get(list.size()-1));
					//System.out.println(connection);
					// points on one third to the end and one third from the start.
					list.add(connection.evaluate(connection.getLastInternalIndex() / 3.0));
					list.add(connection.evaluate((2.0 * connection.getLastInternalIndex()) / 3.0));
					for (PointND p : list) {
						BSplineIntersector intersection = new BSplineIntersector(line, this);
						FunctionOptimizer optimizer = new FunctionOptimizer(2);
						optimizer.setInitialX(computeInitialUV(p));
						optimizer.optimizeFunction(intersection);
						double [] uvvector = optimizer.getOptimum();
						PointND hit = evaluate(Math.exp(uvvector[0]), Math.exp(uvvector[1]));
						if (!intersectionPoints.contains(hit)){
							intersectionPoints.add(hit);
						}
					}
				} else {
					intersectionPoints = list;
				}
				return intersectionPoints;
			} else {
				return null;
			}
		} else 
			throw new RuntimeException("Intersection between BSplineSurfaces and other AbstractCurves are not yet implemented.");
	}

	public PointND [] intersectDeCasteljau(StraightLine line){
		BSplineIntersector intersection = new BSplineIntersector(line, this);
		// TODO
		throw new UnsupportedOperationException("This method is not implemented yet. See Nishita, Sederberg, Kakimoto. Ray Tracing Rational Surface Patches. Computer Graphics, 1990. 24(4):337-45.");


	}

	/**
	 * Tesselates the BSplineSurface into a mesh of Triangles. The parameters samplingU and samplingV define the number of points in each of the internal dimensions. Based on these points a triangle surface mesh is build. The resulting triangles are stored in a CompoundShape for each neighboring point pair in u direction. The u-triangle rings are then put into another CompoundShape. This configuration was optimal when ray tracing in x direction using XCAT BSplines. 
	 * @param samplingU number of points in u direction
	 * @param samplingV number of points in v direction
	 * @return the tesselated mesh
	 */
	public AbstractShape tessellateMesh(double samplingU, double samplingV){
		AbstractShape s = tessellateMesh(samplingU, samplingV, TESSELATE_COMPOUND_OF_COMPOUND_SHAPES);
		s.setName(getTitle());
		return s;
	}
	
	/**
	 * Tesselates the BSplineSurface into a mesh of Triangles. The parameters samplingU and samplingV define the number of points in each of the internal dimensions. Based on these points a triangle surface mesh is build. The resulting triangles are stored in a CompoundShape for each neighboring point pair in u direction. The mode determines the internal structure of the tesselated mesh.   
	 * @param samplingU number of points in u direction
	 * @param samplingV number of points in v direction
	 * @param mode the internal representation of the mesh. Possible modes are
	 * <li>TESSELATE_COMPOUND_SHAPE</li>
	 * <li>TESSELATE_COMPOUND_OF_COMPOUND_SHAPES (default)</li> 
	 * <li>TESSELATE_COMPOUND_OF_OCTREES</li>
	 * <li>TESSELATE_LINEAR_OCTREE</li>
	 * <li>TESSELATE_NESTED_OCTREE</li>
	 * @return the tesselated mesh
	 */
	public AbstractShape tessellateMesh(double samplingU, double samplingV, int mode){
		switch (mode){
			case TESSELATE_COMPOUND_SHAPE: return tessellateMeshCompoundShape(samplingU, samplingV);
			case TESSELATE_COMPOUND_OF_COMPOUND_SHAPES: return tessellateMeshNestedCompoundShapes(samplingU, samplingV);
			case TESSELATE_COMPOUND_OF_OCTREES: return tessellateMeshWithCompundShapesAndLinearOctrees(samplingU, samplingV);
			case TESSELATE_LINEAR_OCTREE: return tessellateMeshLinearOctree(samplingU, samplingV);
			case TESSELATE_NESTED_OCTREE: return tessellateMeshNestedOctree(samplingU, samplingV);
			default: return tessellateMesh(samplingU, samplingV);
		}
	}
	
	private AbstractShape tessellateMeshCompoundShape(double samplingU, double samplingV){
		PointND [] pts = getRasterPoints(samplingU, samplingV);
		CompoundShape superShape = new CompoundShape();
		CompoundShape mesh = superShape;
		ArrayList<PointND> lastSlice = new ArrayList<PointND>();
		for(double i =1 ; i < samplingU; i++){
			double lasti = i - 1;
			if (lasti < 0) lasti = samplingU - 1;
			for (double j = 0; j < samplingV; j++){
				double lastj = j - 1;
				if (lastj < 0) lastj = samplingV - 1;
				if (i == samplingU-1) lastSlice.add(pts[(int)(i*samplingV+j)]);
				try{
					Triangle t1 = new Triangle(pts[(int)(lasti*samplingV+lastj)] ,
							pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)]);
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
					mesh.add(t2);
				} catch (Exception e){
					if (!e.getLocalizedMessage().contains("direction vector")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
			}
		}
		if (lastSlice.size() > 0) {
			PointND center = General.getGeometricCenter(lastSlice);
			for (int i = 1; i < lastSlice.size(); i++){
				try{
					mesh.add(new Triangle(center, lastSlice.get(i), lastSlice.get(i-1)));
				} catch (Exception e){
					if (!e.getLocalizedMessage().contains("direction vector")){
						System.out.println(e.getLocalizedMessage());
					}
				}
			}
			try{
				mesh.add(new Triangle(center, lastSlice.get(0), lastSlice.get(lastSlice.size()-1)));
			} catch (Exception e){
				if (!e.getLocalizedMessage().contains("direction vector")){
					System.out.println(e.getLocalizedMessage());
				}
			}
		}
		System.out.println("Triangles: " + superShape + " " + superShape.getInternalDimension() + " " + superShape.size());
		return superShape;
	}
	
	private AbstractShape tessellateMeshLinearOctree(double samplingU, double samplingV){
		PointND [] pts = getRasterPoints(samplingU, samplingV);
		CompoundShape superShape = new LinearOctree(min, max);
		CompoundShape mesh = superShape;
		ArrayList<PointND> lastSlice = new ArrayList<PointND>();
		for(double i =1 ; i < samplingU; i++){
			double lasti = i - 1;
			if (lasti < 0) lasti = samplingU - 1;
			for (double j = 0; j < samplingV; j++){
				double lastj = j - 1;
				if (lastj < 0) lastj = samplingV - 1;
				if (i == samplingU-1) lastSlice.add(pts[(int)(i*samplingV+j)]);
				try{
					Triangle t1 = new Triangle(pts[(int)(lasti*samplingV+lastj)] ,
							pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)]);
					mesh.add(t1);
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
				try{

					Triangle t2 = new Triangle(pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)] ,
							pts[(int)(i*samplingV+j)]);
					mesh.add(t2);
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
			}
		}
		if (lastSlice.size() > 0) {
			PointND center = General.getGeometricCenter(lastSlice);
			for (int i = 1; i < lastSlice.size(); i++){
				try{
					mesh.add(new Triangle(center, lastSlice.get(i), lastSlice.get(i-1)));
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage());
					}
				}
			}
			try{
				mesh.add(new Triangle(center, lastSlice.get(0), lastSlice.get(lastSlice.size()-1)));
			} catch (Exception e){
				if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
					System.out.println(e.getLocalizedMessage());
				}
			}
		}
		System.out.println("Triangles: " + superShape + " " + superShape.getInternalDimension() + " " + superShape.size());
		return superShape;
	}

	private AbstractShape tessellateMeshNestedOctree(double samplingU, double samplingV){
		PointND [] pts = getRasterPoints(samplingU, samplingV);
		CompoundShape superShape = new NestedOctree(min, max);
		CompoundShape mesh = superShape;
		ArrayList<PointND> lastSlice = new ArrayList<PointND>();
		for(double i =1 ; i < samplingU; i++){
			double lasti = i - 1;
			if (lasti < 0) lasti = samplingU - 1;
			for (double j = 0; j < samplingV; j++){
				double lastj = j - 1;
				if (lastj < 0) lastj = samplingV - 1;
				if (i == samplingU-1) lastSlice.add(pts[(int)(i*samplingV+j)]);
				try{
					Triangle t1 = new Triangle(pts[(int)(lasti*samplingV+lastj)] ,
							pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)]);
					mesh.add(t1);
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
				try{

					Triangle t2 = new Triangle(pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)] ,
							pts[(int)(i*samplingV+j)]);
					mesh.add(t2);
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
			}
		}
		if (lastSlice.size() > 0) {
			PointND center = General.getGeometricCenter(lastSlice);
			for (int i = 1; i < lastSlice.size(); i++){
				try{
					mesh.add(new Triangle(center, lastSlice.get(i), lastSlice.get(i-1)));
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage());
					}
				}
			}
			try{
				mesh.add(new Triangle(center, lastSlice.get(0), lastSlice.get(lastSlice.size()-1)));
			} catch (Exception e){
				if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
					System.out.println(e.getLocalizedMessage());
				}
			}
		}
		mesh.getMin();
		System.out.println("Triangles: " + superShape + " " + superShape.getInternalDimension() + " " + superShape.size());
		return superShape;
	}
	
	private AbstractShape tessellateMeshNestedCompoundShapes(double samplingU, double samplingV){
		PointND [] pts = getRasterPoints(samplingU, samplingV);
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
					mesh.add(t2);
				} catch (Exception e){
					if (!e.getLocalizedMessage().contains("direction vector")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
			}
			superShape.add(mesh);
		}
		superShape.addAll(General.createTrianglesFromPlanarPointSet(lastSlice));
		superShape.addAll(General.createTrianglesFromPlanarPointSet(firstSlice));
		//System.out.println("Triangles: " + superShape + " " + superShape.getInternalDimension() + " " + superShape.size());
		return superShape;
	}
	
	
	
	private AbstractShape tessellateMeshWithCompundShapesAndLinearOctrees(double samplingU, double samplingV){
		PointND [] pts = getRasterPoints(samplingU, samplingV);
		CompoundShape superShape = new CompoundShape();
		ArrayList<PointND> lastSlice = new ArrayList<PointND>();
		for(double i =1 ; i < samplingU; i++){
			CompoundShape mesh =  new CompoundShape();
			double lasti = i - 1;
			if (lasti < 0) lasti = samplingU - 1;
			for (double j = 0; j < samplingV; j++){
				double lastj = j - 1;
				if (lastj < 0) lastj = samplingV - 1;
				if (i == samplingU-1) lastSlice.add(pts[(int)(i*samplingV+j)]);
				try{
					Triangle t1 = new Triangle(pts[(int)(lasti*samplingV+lastj)] ,
							pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)]);
					mesh.add(t1);
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
				try{

					Triangle t2 = new Triangle(pts[(int)(i*samplingV+lastj)] ,
							pts[(int)(lasti*samplingV+j)] ,
							pts[(int)(i*samplingV+j)]);
					mesh.add(t2);
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage() +" "+ i + " " + j + " " + lasti + " " + lastj);
					}
				}
			}
			LinearOctree tree = new LinearOctree(mesh.getMin(), mesh.getMax());
			tree.addAll(mesh);
			System.out.println(tree);
			superShape.add(tree);
		}
		if (lastSlice.size() > 0) {
			CompoundShape mesh = new CompoundShape();
			PointND center = General.getGeometricCenter(lastSlice);
			for (int i = 1; i < lastSlice.size(); i++){
				try{
					mesh.add(new Triangle(center, lastSlice.get(i), lastSlice.get(i-1)));
				} catch (Exception e){
					if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
						System.out.println(e.getLocalizedMessage());
					}
				}
			}
			try{
				mesh.add(new Triangle(center, lastSlice.get(0), lastSlice.get(lastSlice.size()-1)));
			} catch (Exception e){
				if (!e.getLocalizedMessage().equals("Second given direction vector must not be null!")){
					System.out.println(e.getLocalizedMessage());
				}
			}
			LinearOctree tree = new LinearOctree(mesh.getMin(), mesh.getMax());
			tree.addAll(mesh);
			System.out.println(tree);
			superShape.add(tree);
		}
		System.out.println("Triangles: " + superShape + " " + superShape.getInternalDimension() + " " + superShape.size());
		return superShape;
	}
	
	@Override
	public void applyTransform(Transform t) {
		ArrayList<PointND> newPoints = new ArrayList<PointND>();
		for (int i =0; i< points.size(); i++){
			newPoints.add(t.transform(points.get(i)));
		}
		points = newPoints;
		init();
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	/**
	 * Class to intersect bsplines with a StraightLine. Uses a procedure similar to Toth if used as OptimizableFunction. Optimization after Nishita will be implemented next.
	 * @author akmaier
	 *
	 */
	private class BSplineIntersector implements OptimizableFunction {

		double [] planeCoeff1, planeCoeff2;
		StraightLine line;
		SurfaceBSpline spline;

		public BSplineIntersector(StraightLine line, SurfaceBSpline spline) {
			this.line = line;
			this.spline = spline;
			SimpleVector e1 = new SimpleVector(line.getDirection().getLen());
			e1.setElementValue(0, 1);
			SimpleVector planeNormal1 = General.crossProduct(line.getDirection(), e1);
			if (planeNormal1.normL2() < 0.1) {
				e1.setElementValue(0, 0);
				e1.setElementValue(1, 1);
				planeNormal1 = General.crossProduct(line.getDirection(), e1);
			}
			planeNormal1.normalizeL2();
			SimpleVector planeNormal2 = General.crossProduct(line.getDirection(),planeNormal1);
			planeNormal2.normalizeL2();
			SimpleVector point = line.getPoint().getAbstractVector();
			planeCoeff1 = new double [4];
			planeCoeff2 = new double [4];
			planeNormal1.copyTo(planeCoeff1);
			planeNormal2.copyTo(planeCoeff2);
			planeCoeff1[3] = - SimpleOperators.multiplyInnerProd(planeNormal1, point);
			planeCoeff2[3] = - SimpleOperators.multiplyInnerProd(planeNormal2, point);
		}

		/**
		 * computes d1 (after Nishita)
		 * @param x
		 * @return
		 */
		public double computeD1 (double [] x){
			double d1 = 0;
			for (int i = 0; i < numberOfUPoints; i++){
				double weight = uSpline.getWeight(x[0], i);
				double d1Part = 0;
				for (int j = 0; j < numberOfVPoints; j++){
					double weight2 = vSplines[i].getWeight(x[1], j);
					d1Part += weight2 * vSplines[i].distance(planeCoeff1, j);
				}
				d1 += weight * d1Part;
			}
			return d1;
		}

		/**
		 * computes d2 (after Nishita)
		 * @param x
		 * @return
		 */
		public double computeD2 (double [] x){
			double d2 = 0;
			for (int i = 0; i < numberOfUPoints; i++){
				double weight = uSpline.getWeight(x[0], i);
				double d2Part = 0;
				for (int j = 0; j < numberOfVPoints; j++){
					double weight2 = vSplines[i].getWeight(x[1], j);
					d2Part += weight2 * vSplines[i].distance(planeCoeff2, j);
				}
				d2 += weight * d2Part;
			}
			return d2;
		}

		public double evaluate_planebased(double[] x, int block) {
			double d1 = 0;
			double d2 = 0;
			for (int i = 0; i < numberOfUPoints; i++){
				double weight = uSpline.getWeight(Math.exp(x[0]), i);
				double d1Part = 0;
				double d2Part = 0;
				for (int j = 0; j < numberOfVPoints; j++){
					double weight2 = vSplines[i].getWeight(Math.exp(x[1]), j);
					d1Part += weight2 * vSplines[i].distance(planeCoeff1, j);
					d2Part += weight2 * vSplines[i].distance(planeCoeff2, j);
				}
				d1 += weight * d1Part;
				d2 += weight * d2Part;
			}
			double revan = Math.pow(d1, 2) + Math.pow(d2, 2);
			return revan;
		}

		@Override
		public double evaluate(double[] x, int block) {
			PointND p = spline.evaluate(Math.exp(x[0]), Math.exp(x[1]));
			double revan = line.computeDistanceTo(p);
			return revan;
		}

		@Override
		public int getNumberOfProcessingBlocks() {
			return 1;
		}

		@Override
		public void setNumberOfProcessingBlocks(int number) {
		}

	}

	@Override
	public PointND[] getRasterPoints(int number) {
		int subNumber = (int) Math.sqrt(number);
		PointND [] array = new PointND[subNumber * subNumber];
		for (double i = 0; i <subNumber ; i++){
			for (double j = 0; j < subNumber; j++){
				array[(int)(i+(j*subNumber))] = evaluate(i/(subNumber-1), j/(subNumber-1));
			}
		}
		return array;
	}

	/**
	 * Binary Representation of a Surface BSpline:
	 * <pre>
	 * type
	 * total size in float values
	 * ID
	 * # of u knots
	 * # of v knots
	 * # of u points
	 * # of v points
	 * u knot vector
	 * v knot vector
	 * u BSpline
	 * v BSplines
	 * </pre>
	 * @return the binary representation for use with OpenCL
	 */
	public float[] getBinaryRepresentation() {
		float type = BSpline.SPLINE2DTO3D;
		int totalsize;
		float uknot = uKnots.getLen();
		float vknot = vKnots.getLen();
		float uPoints = this.numberOfUPoints;
		float vPoints = this.numberOfVPoints;
		float [] uSpline = this.uSpline.getBinaryRepresentation();
		float [][] vSplines = new float [this.vSplines.length][];
		int vSplinesLength = 0;
		for (int i = 0; i< vSplines.length; i++){
			vSplines[i] = this.vSplines[i].getBinaryRepresentation();
			vSplinesLength += vSplines[i].length;
		}
		totalsize = 7 + uSpline.length + vSplines.length + uKnots.getLen() + vKnots.getLen();
		float [] binary = new float [totalsize];
		binary[0] = type;
		binary[1] = totalsize;
		binary[2] = BSpline.getID(title);
		binary[3] = uknot;
		binary[4] = vknot;
		binary[5] = uPoints;
		binary[6] = vPoints;
		int index =7;
		for (int i=0; i<uknot; i++){
			binary[index] = (float)uKnots.getElement(i);
			index ++;
		}
		for (int i=0; i<vknot; i++){
			binary[index] = (float)vKnots.getElement(i);
			index ++;
		}
		for (int i=0; i<uSpline.length; i++){
			binary[index] = uSpline[i];
			index ++;
		}
		for (int i=0; i<vSplines.length; i++){
			for (int j = 0; j< vSplines[i].length; j++){
				binary[index] = vSplines[i][j];
				index ++;
			}
		}
		return binary;
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		return this.tessellateMesh(100.0 / accuracy, 100.0 / accuracy);
	}

	/**
	 * @return the numberOfUPoints
	 */
	public int getNumberOfUPoints() {
		return numberOfUPoints;
	}

	/**
	 * @param numberOfUPoints the numberOfUPoints to set
	 */
	public void setNumberOfUPoints(int numberOfUPoints) {
		this.numberOfUPoints = numberOfUPoints;
	}

	/**
	 * @return the numberOfVPoints
	 */
	public int getNumberOfVPoints() {
		return numberOfVPoints;
	}

	/**
	 * @param numberOfVPoints the numberOfVPoints to set
	 */
	public void setNumberOfVPoints(int numberOfVPoints) {
		this.numberOfVPoints = numberOfVPoints;
	}

	public PointND[] getRasterPoints(double samplingU, double samplingV) {
		PointND [] pts = new PointND[((int)samplingU) * ((int)samplingV)];
		for(double i =0 ; i < samplingU; i++){
			for (double j = 0; j < samplingV; j++){
				PointND p = evaluate(i/samplingU, j/ samplingV);
				pts[(int)(i*samplingV+j)] = p;
			}
		}
		return pts;
	}

	@Override
	public AbstractShape clone() {
		return new SurfaceBSpline(this);
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/