/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.io;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class PlyIO {
	
	public void write(String filename, ArrayList<PointND> points){
		System.out.println("Writing to file: " + filename);
		File f = new File(filename);
		f.getParentFile().mkdirs();
		
		try{
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			writer.println("ply");
			writer.println("format ascii 1.0");
			writer.println("comment VCGLIB generated");
			writer.println("element vertex " + String.valueOf(points.size()));
			writer.println("property float x");
			writer.println("property float y");
			writer.println("property float z");
			writer.println("element face 0");
			writer.println("property list uchar int vertex_indices");
			writer.println("end_header");
			for(int i = 0; i < points.size(); i++){
				PointND p = points.get(i);
				String line = "";
				for(int j = 0; j < 3; j++){
					line += String.valueOf(p.get(j)) + " ";
				}
				writer.println(line);
			}
			writer.close();
			System.out.println("Done.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
		
	
	//TODO read method
			
}
