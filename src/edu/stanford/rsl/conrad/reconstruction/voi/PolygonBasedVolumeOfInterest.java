package edu.stanford.rsl.conrad.reconstruction.voi;

import java.awt.Polygon;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import edu.stanford.rsl.conrad.io.ConfigFileParser;
import edu.stanford.rsl.conrad.io.SafeSerializable;

/**
 * VOI based on a polygon definition which is identical for each slice. The VOI is than formed as a stack of identical polygons. Note that this method is sub-optimal for clipping as either the VOI has to be pre-computed for the whole volume or clipped against a polygon in each call of contains(). 
 * @author akmaier
 *
 */
public class PolygonBasedVolumeOfInterest extends VolumeOfInterest implements ConfigFileParser, SafeSerializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8511498569857161802L;
	Polygon sliceRegionOfInterest;
	double maxZ, minZ;
	int numberOfPointsInPolygon;
	private boolean success = false;

	private HashMap<Double, Boolean> sliceVoi = new HashMap<Double, Boolean>();

	public PolygonBasedVolumeOfInterest(){
	}

	public PolygonBasedVolumeOfInterest(String maxVOIFileName) throws IOException{
		readConfigFile(maxVOIFileName);
	}

	
	
	public boolean contains (double x, double y, double z){
		if (debug) System.out.println("Voxel " + x + " " + y + " " + z);
		boolean revan = true;
		if ((z < minZ)||(z > maxZ)){
			revan = false;
			if (debug) System.out.println("z out of Range: " + z);
			if (debug) System.out.println("Z-Range " + minZ + " " + maxZ);
		} else {
			Double location = new Double (x *1000 +y);
			if (! sliceVoi.containsKey(location)) {
				if (!(sliceRegionOfInterest.contains(x, y))){
					revan = false;
					if (debug) System.out.println("Not in polygon");
				}
				sliceVoi.put(location, new Boolean(revan));
			} else {
				revan = sliceVoi.get(location);
			}
		}
		return revan;
	}

	public void readConfigFile(String filename) throws IOException {
		FileReader read = new FileReader(filename);
		BufferedReader bufferedReader = new BufferedReader(read);
		String line = "";
		//skip four lines
		line = bufferedReader.readLine();
		if (line.contains("version 2")){
			line = bufferedReader.readLine();
			line = bufferedReader.readLine();
			line = bufferedReader.readLine();
			if (debug) System.out.println(line);
			String [] entries = line.split("\\s+");
			minZ = Double.parseDouble(entries[1]);
			maxZ = Double.parseDouble(entries[2]);
			if (debug) System.out.println("Z-Range " + minZ + " " + maxZ);
			line = bufferedReader.readLine();
			if (debug) System.out.println(line);
			entries = line.split("\\s+");
			numberOfPointsInPolygon = Integer.parseInt(entries[1]);
			int [] xvals = new int[numberOfPointsInPolygon];
			int [] yvals = new int[numberOfPointsInPolygon];
			for (int i=0; i < numberOfPointsInPolygon; i++){
				line = bufferedReader.readLine();
				entries = line.split("\\s+");
				xvals[i] = (int) Math.round(Double.parseDouble(entries[1]));
				yvals[i] = (int) Math.round(Double.parseDouble(entries[2]));
				if (debug) System.out.println(line);
			}
			sliceRegionOfInterest = new Polygon(xvals, yvals, numberOfPointsInPolygon);
			success = true;
		} else {
			throw new IOException("File is not in version 2 format.");
		}
		bufferedReader.close();
	}

	public boolean getSuccess() {
		return success;
	}

	public void prepareForSerialization() {
		sliceVoi = new HashMap<Double, Boolean>();
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/