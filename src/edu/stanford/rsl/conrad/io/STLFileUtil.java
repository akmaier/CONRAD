/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.TriangleMesh;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.MTFBeadPhantom;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.FileUtil;

public class STLFileUtil {


	public static void writeToSTLFile(String filename, PrioritizableScene scene) throws IOException{
		// Open file to write

		FileWriter fw = new FileWriter(new File(filename));
		BufferedWriter bw = new BufferedWriter(fw);

		for (PhysicalObject o: scene){
			writePhysicalObject(bw, o);
		}
		bw.flush();
		bw.close();

	}

	private static void writePhysicalObject(BufferedWriter bw, PhysicalObject o) throws IOException{
		bw.write("solid " + o.getNameString() + "\r\n");
		writeShape(bw, o.getShape());
		bw.write("endsolid "+ o.getNameString() + "\r\n");

	}

	private static void writeShape(BufferedWriter bw, AbstractShape shape) throws IOException{
		if (shape instanceof CompoundShape){
			for (AbstractShape s2: (CompoundShape)shape){
				writeShape(bw, s2);
			}
		}
		if (shape instanceof Triangle){
			Triangle tri = (Triangle) shape;
			bw.write("  facet normal " + vectorToString(tri.getNormal())+"\r\n");
			bw.write("  outer loop\r\n");
			bw.write("    vertex " + vectorToString(tri.getA().getAbstractVector())+"\r\n");
			bw.write("    vertex " + vectorToString(tri.getB().getAbstractVector())+"\r\n");
			bw.write("    vertex " + vectorToString(tri.getC().getAbstractVector())+"\r\n");
			bw.write("  endloop\r\n");
			bw.write("  endfacet\r\n");
		}
	}

	private static String vectorToString(SimpleVector vec){
		return "" + vec.getElement(0) + " " + vec.getElement(1) + " " + vec.getElement(2);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		boolean testWrite = false;
		if (testWrite){
			AnalyticPhantom phantom = new MTFBeadPhantom();
			try {
				String filename = FileUtil.myFileChoose(".stl", true);
				STLFileUtil.writeToSTLFile(filename, phantom.tessellatePhantom(0.5));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			try {
				CompoundShape mesh = readSTLMesh(FileUtil.myFileChoose(".stl", false));

				System.out.println("Read mesh with " + mesh.size() + " triangles");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	/**
	 * Method to read an ascii STL Mesh from a file.
	 * @param filename the filename
	 * @return the mesh
	 * @throws IOException may occur
	 */
	public static CompoundShape readSTLMesh(String filename) throws IOException{
		return readSTLMesh(filename, 100);
	}

	public static CompoundShape readSTLMesh(String filename, int subCompoundStep) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String lineString = br.readLine();
		CompoundShape mesh = new TriangleMesh();
		if (lineString.startsWith("solid")){
			int currentCount = 0;
			CompoundShape currentShape = new CompoundShape();
			SimpleVector normal = null;
			// read solid
			lineString = br.readLine();
			while (lineString !=null) {
				if (lineString.contains("facet normal")){
					normal = readVector(lineString);
					lineString = br.readLine();
				}
				if (lineString.contains("outer loop")){
					PointND one = new PointND(readVector(br.readLine()));
					PointND two = new PointND(readVector(br.readLine()));
					PointND three = new PointND(readVector(br.readLine()));
					Triangle tri = new Triangle(one, two, three);
					double test = SimpleOperators.multiplyInnerProd(normal, tri.getNormal());
					if (test<0) tri.flipNormal();
					currentShape.add(tri);
					currentCount++;
					if (currentCount >= subCompoundStep){
						currentCount = 0;
						mesh.add(currentShape);
						currentShape = new CompoundShape();
					}
					lineString = br.readLine();
					if (!lineString.contains("endloop")) throw new RuntimeException("Mesh did not contain triangular data!");
					lineString = br.readLine();
				}
				if (!lineString.contains("endfacet")) throw new RuntimeException("Malformed facet detected!");
				lineString = br.readLine();
				if (lineString.contains("endsolid")) break;
			}
			if (currentShape.size() > 0) mesh.add(currentShape);
		}
		mesh.getMax();
		mesh.getMin();
		return mesh;
	}

	public static SimpleVector readVector(String lineString){
		String [] substr = lineString.split("\\s+");
		double one = Double.parseDouble(substr[substr.length-3]);
		double two = Double.parseDouble(substr[substr.length-2]);
		double three = Double.parseDouble(substr[substr.length-1]);
		return new SimpleVector(one, two, three);
	}


}
