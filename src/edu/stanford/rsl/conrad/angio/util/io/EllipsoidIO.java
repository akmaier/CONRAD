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

import edu.stanford.rsl.conrad.geometry.shapes.simple.Ellipsoid;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class EllipsoidIO {
	
	private ArrayList<PointND> centers = null;
	
	public static void main(String[] args){
		ArrayList<Ellipsoid> test = new ArrayList<Ellipsoid>();
		test.add(new Ellipsoid(1,2,3,new AffineTransform(SimpleMatrix.I_3, new SimpleVector(4,5,6))));
		EllipsoidIO io = new EllipsoidIO();
		
		String filename = ".../recon/test.ellipsoids";
		
		io.write(filename, test);
	}
	
	public ArrayList<Ellipsoid> read(String filename){
		System.out.println("Reading from file: " + filename);
		ArrayList<Ellipsoid> el = new ArrayList<Ellipsoid>();
		this.centers = new ArrayList<PointND>();
		
		FileReader fr;
		try {
			fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Number of ellipsoids"
			tok.nextToken();
			tok.nextToken();
			String t = tok.nextToken();
			int size = Integer.parseInt(t);
			for(int i = 0; i < size; i++){
				line = br.readLine();
				tok = new StringTokenizer(line);
				double[] point = new double[3];
				for(int j = 0; j < 3; j++){
					point[j] = Double.parseDouble(tok.nextToken());
				}
				double[] dxyz = new double[3];
				for(int j = 0; j < 3; j++){
					dxyz[j] = Double.parseDouble(tok.nextToken());
				}
				centers.add(new PointND(point));
				AffineTransform aff = new AffineTransform(SimpleMatrix.I_3, new SimpleVector(point));
				Ellipsoid e = new Ellipsoid(dxyz[0], dxyz[1], dxyz[2], aff);
				el.add(e);
			}
		
			br.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return el;
	}
	
	public ArrayList<PointND> getCenters() {
		return centers;
	}

	public void write(String filename, ArrayList<Ellipsoid> el){
		System.out.println("Writing to file: " + filename);
		File f = new File(filename);
		f.getParentFile().mkdirs();
		
		try{
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			
			writer.println("Number of ellipsoids: " + Integer.valueOf(el.size()));
			for(int i = 0; i < el.size(); i++){
				Ellipsoid e = el.get(i);
				Transform t = e.getTransform();
				String line = "";
				for(int j = 0; j < 3; j++){
					line += Double.valueOf(t.getTranslation(3).getElement(j)) + " ";
				}
				line = line + Double.valueOf(e.dx) + " " + Double.valueOf(e.dy) + " "+ Double.valueOf(e.dz);
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

