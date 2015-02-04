package edu.stanford.rsl.conrad.phantom.xcat;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

/**
 * file handling of VICON marker txt file
 * VICON txt files need to be in the XCAT folder.
 * 
 * @author Jang CHOI 
 */

public class ViconMarkerBuilder implements Serializable  {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2023706815633141182L;
	private static String XCatDirectory;
	private static File fileMarkers;
	
	private static String measurementMode = "Static60"; // Static40, Dynamic, Static60
	//private static String subject = "Subject 5"; // The smallest;	
	private static String subject = "Subject 2"; // The biggest;
	
	// column number for each item
	private static int colNoLKneeAngleX, colNoRKneeAngleX, 
		colNoLHJCX, colNoLHJCY, colNoLHJCZ, 
		colNoRHJCX, colNoRHJCY, colNoRHJCZ, 
		colNoLKJCX,	colNoLKJCY, colNoLKJCZ, 
		colNoRKJCX, colNoRKJCY, colNoRKJCZ, 
		colNoLAJCX, colNoLAJCY, colNoLAJCZ, 
		colNoRAJCX, colNoRAJCY, colNoRAJCZ,
		colNoLKNEX,	colNoLKNEY, colNoLKNEZ,
		colNoRKNEX,	colNoRKNEY, colNoRKNEZ,
		colNoLPATX,	colNoLPATY, colNoLPATZ,
		colNoRPATX,	colNoRPATY, colNoRPATZ;
	
	private ArrayList<ArrayList<Double>> markers;
	
	public ViconMarkerBuilder(){
		markers = rebuildMarkerFromFile();
	}
		
			
	public ArrayList<ArrayList<Double>> rebuildMarkerFromFile() {
		
		ArrayList<ArrayList<Double>> myMarkers = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> myCurrentRow;

		try {
			XCatDirectory = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.XCAT_PATH);
			//XCatDirectory = "D:\\Data\\WeightBearing\\stanford_knee_data_jang_2013_07_08";
			String filename = XCatDirectory + "\\VICON\\" + subject + "\\" + measurementMode + ".txt";
			fileMarkers = new File(filename);
						
			// create BufferedReader to read txt file
			BufferedReader br = new BufferedReader(new FileReader(fileMarkers));
			String strLine = "";
			StringTokenizer st = null;
			int lineNumber = 0, tokenNumber = 0;
			String tokenName;
			
			//read tab separated file line by line
			while( (strLine = br.readLine()) != null)
			{
				//break tab separated line using "\t"
				st = new StringTokenizer(strLine, "\t");
				myCurrentRow = new ArrayList<Double>(); 
				
				while(st.hasMoreTokens()){
					
					tokenName = st.nextToken().trim(); 
					if (lineNumber == 0){
						if (tokenName.equals("LKneeAngle:X")) colNoLKneeAngleX = tokenNumber;
						if (tokenName.equals("RKneeAngle:X")) colNoRKneeAngleX = tokenNumber;
						if (tokenName.equals("LHJC:X")) colNoLHJCX = tokenNumber;
						if (tokenName.equals("LHJC:Y")) colNoLHJCY = tokenNumber;
						if (tokenName.equals("LHJC:Z")) colNoLHJCZ = tokenNumber;
						if (tokenName.equals("RHJC:X")) colNoRHJCX = tokenNumber;
						if (tokenName.equals("RHJC:Y")) colNoRHJCY = tokenNumber;
						if (tokenName.equals("RHJC:Z")) colNoRHJCZ = tokenNumber;
						if (tokenName.equals("LKJC:X")) colNoLKJCX = tokenNumber;
						if (tokenName.equals("LKJC:Y")) colNoLKJCY = tokenNumber;
						if (tokenName.equals("LKJC:Z")) colNoLKJCZ = tokenNumber;
						if (tokenName.equals("RKJC:X")) colNoRKJCX = tokenNumber;
						if (tokenName.equals("RKJC:Y")) colNoRKJCY = tokenNumber;
						if (tokenName.equals("RKJC:Z")) colNoRKJCZ = tokenNumber;
						if (tokenName.equals("LAJC:X")) colNoLAJCX = tokenNumber;
						if (tokenName.equals("LAJC:Y")) colNoLAJCY = tokenNumber;
						if (tokenName.equals("LAJC:Z")) colNoLAJCZ = tokenNumber;
						if (tokenName.equals("RAJC:X")) colNoRAJCX = tokenNumber;
						if (tokenName.equals("RAJC:Y")) colNoRAJCY = tokenNumber;
						if (tokenName.equals("RAJC:Z")) colNoRAJCZ = tokenNumber;
						if (tokenName.equals("LKNE:X")) colNoLKNEX = tokenNumber;
						if (tokenName.equals("LKNE:Y")) colNoLKNEY = tokenNumber;
						if (tokenName.equals("LKNE:Z")) colNoLKNEZ = tokenNumber;
						if (tokenName.equals("RKNE:X")) colNoRKNEX = tokenNumber;
						if (tokenName.equals("RKNE:Y")) colNoRKNEY = tokenNumber;
						if (tokenName.equals("RKNE:Z")) colNoRKNEZ = tokenNumber;
						if (tokenName.equals("LPAT:X")) colNoLPATX = tokenNumber;
						if (tokenName.equals("LPAT:Y")) colNoLPATY = tokenNumber;
						if (tokenName.equals("LPAT:Z")) colNoLPATZ = tokenNumber;
						if (tokenName.equals("RPAT:X")) colNoRPATX = tokenNumber;
						if (tokenName.equals("RPAT:Y")) colNoRPATY = tokenNumber;
						if (tokenName.equals("RPAT:Z")) colNoRPATZ = tokenNumber;
						
					} else {	
						myCurrentRow.add(Double.valueOf(tokenName).doubleValue());
					}
					
					//display txt values					
					//System.out.println("Line # " + lineNumber + ", Token # " + tokenNumber + ", Token : " + tokenName);
					tokenNumber++;
				}
				
				// save except the first title row
				if(lineNumber > 0) myMarkers.add(myCurrentRow);
				
				lineNumber++;
				//reset token number
				tokenNumber = 0;
			}
			
			
		} catch(Exception e) {
			System.out.println("Exception while reading txt file: " + e);
		}
		return myMarkers;
	}
	
	public ArrayList<ArrayList<Double>> getVICONMarkers() {
		return markers;
	}
	
	public int getColNoLKneeAngleX(){
		return colNoLKneeAngleX;
	}	
	public int getColNoRKneeAngleX(){
		return colNoRKneeAngleX;
	}
	public int colNoLHJCX(){
		return colNoLHJCX;
	}
	public int colNoLHJCY(){
		return colNoLHJCY;
	}
	public int colNoLHJCZ(){
		return colNoLHJCZ;
	}
	public int colNoRHJCX(){
		return colNoRHJCX;
	}
	public int colNoRHJCY(){
		return colNoRHJCY;
	}
	public int colNoRHJCZ(){
		return colNoRHJCZ;
	}
	public int colNoLKJCX(){
		return colNoLKJCX;
	}
	public int colNoLKJCY(){
		return colNoLKJCY;
	}
	public int colNoLKJCZ(){
		return colNoLKJCZ;
	}
	public int colNoRKJCX(){
		return colNoRKJCX;
	}
	public int colNoRKJCY(){
		return colNoRKJCY;
	}
	public int colNoRKJCZ(){
		return colNoRKJCZ;
	}
	public int colNoLAJCX(){
		return colNoLAJCX;
	}
	public int colNoLAJCY(){
		return colNoLAJCY;
	}
	public int colNoLAJCZ(){
		return colNoLAJCZ;
	}
	public int colNoRAJCX(){
		return colNoRAJCX;
	}
	public int colNoRAJCY(){
		return colNoRAJCY;
	}
	public int colNoRAJCZ(){
		return colNoRAJCZ;
	}
	public int colNoLKNEX(){
		return colNoLKNEX;
	}
	public int colNoLKNEY(){
		return colNoLKNEY;
	}
	public int colNoLKNEZ(){
		return colNoLKNEZ;
	}
	public int colNoRKNEX(){
		return colNoRKNEX;
	}
	public int colNoRKNEY(){
		return colNoRKNEY;
	}
	public int colNoRKNEZ(){
		return colNoRKNEZ;
	}
	public int colNoLKneeAngleX(){
		return colNoLKneeAngleX;
	}
	public int colNoRKneeAngleX(){
		return colNoRKneeAngleX;
	}
	public int colNoLPATX(){
		return colNoLPATX;
	}
	public int colNoLPATY(){
		return colNoLPATY;
	}
	public int colNoLPATZ(){
		return colNoLPATZ;
	}
	public int colNoRPATX(){
		return colNoRPATX;
	}
	public int colNoRPATY(){
		return colNoRPATY;
	}
	public int colNoRPATZ(){
		return colNoRPATZ;
	}
}
/*
 * Copyright (C) 2010-2014 Jang Hwan Choi
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
