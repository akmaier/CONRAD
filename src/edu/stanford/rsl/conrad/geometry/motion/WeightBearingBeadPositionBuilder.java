package edu.stanford.rsl.conrad.geometry.motion;

import java.io.BufferedWriter;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

/**
 * File handling of beads marker txt file (INITIAL_BEADS_LOCATION_FILE in RegKeys).
 * This is used for the weight-bearing project
 * 
 * @author Jang-Hwan Choi 
 */

public class WeightBearingBeadPositionBuilder implements Serializable  {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -4691626378724299348L;
		
	private static String beadInitialPositionFile;
	private static File fileMarkers;
	
	// column number for each item
	// private static int colNoProjectionNo, colNoBeadNo, colNoU, colNoV;
	
	Configuration config = Configuration.getGlobalConfiguration();
	
	int noProjection = config.getGeometry().getProjectionStackSize();
	public static int beadNo = 22; // total bead number attached on both knee
	public static int currentBeadNo = 7;
	
	
	//private static ArrayList<ArrayList<Double>> beads;
	private double [][][] beadsIn2D = new double [noProjection][beadNo][3]; // [projection #][bead #][u, v, state[0: initial, 1: registered, 2: detected by hough seraching]]
	private double [][] beadsMeanIn3D = new double [beadNo][3]; ; // [bead #][x, y, z]
	
	public WeightBearingBeadPositionBuilder(){
		//beadsIn2D = readInitialBeadPositionFromFile();
	}
		
	public static void main(String [] args) {
		
		CONRAD.setup();
		WeightBearingBeadPositionBuilder beadBuilder = new WeightBearingBeadPositionBuilder();
		beadBuilder.readInitialBeadPositionFromFile();
		beadBuilder.estimateBeadMeanPositionIn3D();
		//beadBuilder.writeBeadPositionToFile();
		
		System.out.println("DONE");
	}
			
	public void readInitialBeadPositionFromFile() {
				 
		ArrayList<Double> myCurrentRow;

		try {
			beadInitialPositionFile = config.getRegistryEntry(RegKeys.INITIAL_BEADS_LOCATION_FILE);			
			fileMarkers = new File(beadInitialPositionFile);

			// create BufferedReader to read txt file
			BufferedReader br = new BufferedReader(new FileReader(fileMarkers));
			String strLine = "";
			StringTokenizer st = null;
			int lineNumber = 0, tokenNumber = 0;
			String tokenName;
			
			//read tab separated file line by line
			while( (strLine = br.readLine()) != null)
			{
				//break tab separated line using " "
				st = new StringTokenizer(strLine, "\t");
				myCurrentRow = new ArrayList<Double>();
				
				while(st.hasMoreTokens()){
					
					tokenName = st.nextToken().trim(); 
					if (lineNumber > 1){
						myCurrentRow.add(Double.valueOf(tokenName).doubleValue());
					}					

					//display txt values					
					//System.out.println("Line # " + lineNumber + ", Token # " + tokenNumber + ", Token : " + tokenName);
					tokenNumber++;
				}
				
				// save except the first 2 title rows
				if(lineNumber > 1) {
					beadsIn2D[myCurrentRow.get(0).intValue()][myCurrentRow.get(1).intValue()][0] = myCurrentRow.get(2);  
					beadsIn2D[myCurrentRow.get(0).intValue()][myCurrentRow.get(1).intValue()][1] = myCurrentRow.get(3);
					if (myCurrentRow.size() < 5)
						beadsIn2D[myCurrentRow.get(0).intValue()][myCurrentRow.get(1).intValue()][2] = 1; // state 1: registered by txt file
					else 
						beadsIn2D[myCurrentRow.get(0).intValue()][myCurrentRow.get(1).intValue()][2] = myCurrentRow.get(4); // state 1: registered by txt file
				}
				
				lineNumber++;
				//reset token number
				tokenNumber = 0;
			}
			
			
		} catch(Exception e) {
			System.out.println("Exception while reading txt file: " + e);
		}
				
	}
	
	public void writeBeadPositionToFile() {

		try {
			fileCopy();
			
			beadInitialPositionFile = config.getRegistryEntry(RegKeys.INITIAL_BEADS_LOCATION_FILE);			
			
			BufferedWriter out = new BufferedWriter(new FileWriter(beadInitialPositionFile));			
			String strLine = DateFormat.getDateTimeInstance(DateFormat.LONG, DateFormat.LONG).format(new Date()) + "\n";			
			strLine += "P_NO\tBEAD_NO\tU\tV\tSTATE[0: initial, 1: registered, 2: detected by hough seraching]\n";
			
			// [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough seraching]]
			double [][][] beadPosition2D = config.getBeadPosition2D();
			int projNo = config.getGeometry().getProjectionStackSize();
			
			//for (int i=0; i< beadNo; i++){				
			for (int i=currentBeadNo; i>= 0; i--){
				for (int j=0; j< projNo; j++) {
					//if (beadPosition2D[j][i][2]>0) {
			
						strLine += j + "\t" + i + "\t" + beadPosition2D[j][i][0] + "\t" + beadPosition2D[j][i][1] + "\t" + (int)beadPosition2D[j][i][2] + "\n";
						
					//}
				}
			}
			
		    out.write(strLine);
		    out.close();
	
		} catch(Exception e) {
			System.out.println("Exception while writing txt file: " + e);
		}
				
	}
			
		
	public void estimateBeadMeanPositionIn3D() {
				
		/*
		 * Muv(u,v) = M * Mxyz  
		 */
		Jama.Matrix Muv; // = new Jama.Matrix (3, 3);
		Jama.Matrix M; // = new Jama.Matrix (3, 1);
		Jama.Matrix Mxyz = new Jama.Matrix (3, 1);		
			
		int beadNoRegistered;
		double u, v, state; 
		SimpleMatrix mat;
		
		for (int i=0; i< beadNo; i++){
			
			beadNoRegistered = 0;		
			
			for (int j=0; j< noProjection; j++) {
				if (beadsIn2D[j][i][0]>0 && beadsIn2D[j][i][1]>0) {
					beadNoRegistered++;
				}
			}
			
			Muv = new Jama.Matrix (2*beadNoRegistered, 1);
			M = new Jama.Matrix (2*beadNoRegistered, 3);
			
			beadNoRegistered = 0;
			for (int j=0; j< noProjection; j++) {
				
				mat = config.getGeometry().getProjectionMatrix(j).computeP();
									
				u = beadsIn2D[j][i][0];
				v = beadsIn2D[j][i][1];
				state = beadsIn2D[j][i][2];
								
				// if initial bead position in 2d is registered, (exclude beads detected by hough)
				if (state == 1) {
					beadNoRegistered++;

					M.set((beadNoRegistered-1)*2, 0, u*mat.getElement(2, 0)-mat.getElement(0, 0));
					M.set((beadNoRegistered-1)*2, 1, u*mat.getElement(2, 1)-mat.getElement(0, 1));
					M.set((beadNoRegistered-1)*2, 2, u*mat.getElement(2, 2)-mat.getElement(0, 2));
					M.set((beadNoRegistered-1)*2+1, 0, v*mat.getElement(2, 0)-mat.getElement(1, 0));
					M.set((beadNoRegistered-1)*2+1, 1, v*mat.getElement(2, 1)-mat.getElement(1, 1));
					M.set((beadNoRegistered-1)*2+1, 2, v*mat.getElement(2, 2)-mat.getElement(1, 2));
					
					Muv.set((beadNoRegistered-1)*2, 0, -u*mat.getElement(2, 3)+mat.getElement(0, 3));
					Muv.set((beadNoRegistered-1)*2+1, 0, -v*mat.getElement(2, 3)+mat.getElement(1, 3));
				}
			}	
			
			if (beadNoRegistered > 3) {
				//Mxyz = M.inverse().times(Muv);
				Mxyz = M.solve(Muv);
				
				beadsMeanIn3D[i][0] = Mxyz.get(0, 0);
				beadsMeanIn3D[i][1] = Mxyz.get(1, 0);
				beadsMeanIn3D[i][2] = Mxyz.get(2, 0);
				
				//System.out.println("x,y,z="+Mxyz.get(0, 0)+","+Mxyz.get(1, 0)+","+Mxyz.get(2, 0));
			}
						
		}
		
		config.setBeadMeanPosition3D(beadsMeanIn3D);
		config.setBeadPosition2D(beadsIn2D);
		
	}
	
	public double [][][] getBeadPositionIn2D() {
		return beadsIn2D;
	}
	
	public void fileCopy() throws IOException {
		
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd_HH-mm-ss");
		
		// file backup		  
		File inputFile = new File(config.getRegistryEntry(RegKeys.INITIAL_BEADS_LOCATION_FILE));
		File outputFile = new File(config.getRegistryEntry(RegKeys.INITIAL_BEADS_LOCATION_FILE) + "_" + format.format(new Date()) + ".txt");
		
		FileReader in = new FileReader(inputFile);
		FileWriter out = new FileWriter(outputFile);
		int c;
		
		while ((c = in.read()) != -1)
		  out.write(c);
		
		in.close();
		out.close();			  
	}
	

}
/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/