/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class EdgeListIO {

	
	public static ArrayList<ArrayList<Edge>> read(String filename){
		System.out.println("Reading from file: " + filename);
		ArrayList<ArrayList<Edge>> edges = new ArrayList<ArrayList<Edge>>();
		
		FileReader fr;
		try {
			fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Number of points"
			ArrayList<Integer> sizes = new ArrayList<Integer>();
			while(tok.hasMoreTokens()){
				int size = Integer.parseInt(tok.nextToken());
				sizes.add(size);
			}
			for(int i = 0; i < sizes.size(); i++){
				ArrayList<Edge> currentEdges = new ArrayList<Edge>();
				for(int j = 0; j < sizes.get(i); j++){
					line = br.readLine();
					tok = new StringTokenizer(line);
					double[] p1 = new double[3];
					double[] p2 = new double[3];
					for(int k = 0; k < 3; k++){
						p1[k] = Double.parseDouble(tok.nextToken());
					}
					PointND pND1 = new PointND(p1[0],p1[1],p1[2]);
					for(int k = 0; k < 3; k++){
						p2[k] = Double.parseDouble(tok.nextToken());
					}
					PointND pND2 = new PointND(p2[0],p2[1],p2[2]);
					currentEdges.add(new Edge(pND1,pND2));
				}
				edges.add(currentEdges);
			}
		
			br.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return edges;
	}
	
	public static void write(String filename, ArrayList<ArrayList<Edge>> edges){
		System.out.println("Writing to file: " + filename);
		File f = new File(filename);
		f.getParentFile().mkdirs();
		
		try{
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			
			String header = "Sizes: ";
			for(int i = 0; i < edges.size(); i++){
				header += String.valueOf(edges.get(i).size())+" ";
			}
			writer.println(header);
			for(int i = 0; i < edges.size(); i++){
				for(int j = 0; j < edges.get(i).size(); j++){
					PointND p1 = edges.get(i).get(j).getPoint();
					PointND p2 = edges.get(i).get(j).getEnd();
					String line = String.valueOf(p1.get(0))+" "+String.valueOf(p1.get(1))+" "+String.valueOf(p1.get(2))+" ";
					line += String.valueOf(p2.get(0))+" "+String.valueOf(p2.get(1))+" "+String.valueOf(p2.get(2));
					writer.println(line);
				}
			}
			writer.close();
			System.out.println("Done.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	
}
