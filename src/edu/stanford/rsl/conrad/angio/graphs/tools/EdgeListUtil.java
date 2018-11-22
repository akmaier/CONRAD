package edu.stanford.rsl.conrad.angio.graphs.tools;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class EdgeListUtil {

	public static ArrayList<PointND> edgeListToPointList(ArrayList<Edge> list){
		ArrayList<PointND> pts = new ArrayList<PointND>();
		for(int i = 0; i < list.size(); i++){
			pts.add(list.get(i).getPoint());
		}
		pts.add(list.get(list.size()-1).getEnd());
		return pts;
	}
	
	public static ArrayList<Edge> pointListToEdgeList(ArrayList<PointND> points){
		ArrayList<Edge> edges = new ArrayList<Edge>();
		for(int i = 0; i < points.size()-1; i++){
			edges.add(new Edge(points.get(i), points.get(i+1)));
		}
		return edges;
	}
	
}
