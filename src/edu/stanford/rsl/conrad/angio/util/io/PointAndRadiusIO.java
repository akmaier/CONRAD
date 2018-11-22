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

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class PointAndRadiusIO {
	private ArrayList<PointND> coords = null;
	private ArrayList<Double> radii = null;
	
	public static void main(String[] args){
		ArrayList<PointND> test = new ArrayList<PointND>();
		test.add(new PointND(1,2,0));
		ArrayList<Double> td = new ArrayList<Double>();
		td.add(2d);
		
		PointAndRadiusIO io = new PointAndRadiusIO();
		
		String filename = ".../recon/test.reco3D";
		
		io.write(filename, test,td);
		
		io.read(filename);
		ArrayList<PointND> read = io.getPoints();
		ArrayList<Double> readL = io.getRadii();
		
		System.out.println(read.size() + " "+ readL.size());
		System.out.println("Done.");
	}
	
	public void read(String filename){
		System.out.println("Reading from file: " + filename);
		this.coords = new ArrayList<PointND>();
		this.radii = new ArrayList<Double>();
				
		FileReader fr;
		try {
			fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Number of points"
			tok.nextToken();
			tok.nextToken();
			String t = tok.nextToken();
			int size = Integer.parseInt(t);
			br.readLine(); // skip explanation line
			for(int i = 0; i < size; i++){
				line = br.readLine();
				tok = new StringTokenizer(line);
				double[] point = new double[3];
				for(int j = 0; j < 3; j++){
					point[j] = Double.parseDouble(tok.nextToken());
				}
				radii.add(Double.parseDouble(tok.nextToken()));
				coords.add(new PointND(point[0], point[1], point[2]));				
			}
		
			br.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public ArrayList<PointND> getPoints() {
		return coords;
	}
	
	public ArrayList<Double> getRadii() {
		return radii;
	}

	public void write(String filename, ArrayList<PointND> points, ArrayList<Double> radii){
		System.out.println("Writing to file: " + filename);
		File f = new File(filename);
		f.getParentFile().mkdirs();
		
		try{
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			
			writer.println("Number of points: " + Integer.valueOf(points.size()));
			writer.println("c_x | c_y | c_z | radius in mm");
			for(int i = 0; i < points.size(); i++){
				PointND p = points.get(i);
				String line = String.valueOf(p.get(0))+" "+String.valueOf(p.get(1))+" "+String.valueOf(p.get(2))+" ";
				line += String.valueOf(radii.get(i));
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

}
