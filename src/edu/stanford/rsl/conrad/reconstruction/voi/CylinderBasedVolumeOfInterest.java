package edu.stanford.rsl.conrad.reconstruction.voi;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Describes a VOI consisting of a cylinder with one cone on top and one cone on the bottom. This is a typical VOI for a C-arm CT acquisition.
 * @author akmaier
 *
 */
public class CylinderBasedVolumeOfInterest extends VolumeOfInterest {

	/**
	 * 
	 */
	private static final long serialVersionUID = 431870065424171642L;
	double radius;
	double minz, maxz;
	double cylinderminz, cylindermaxz;
	private boolean success = false;
	@Override
	public boolean contains(double x, double y, double z) {
		boolean contains = false;
		if ((z < cylindermaxz) && (z > cylinderminz)){
			contains = inRadius(x, y, radius);
		} else {
			if ((z > cylindermaxz) && (z < maxz)){
				double maxrange = maxz - cylindermaxz;
				double fraction = 1.0  - ((z - cylindermaxz) / maxrange);
				contains = inRadius(x, y, radius * fraction);
			} else {
				if ((z < cylinderminz) && (z > minz)){
					double maxrange = cylinderminz - minz;
					double fraction = 1.0  - ((cylinderminz - z) / maxrange);
					contains = inRadius(x, y, radius * fraction);
				}
			}
		}
		return contains;
	}
	
	private boolean inRadius(double x, double y, double radius){
		double sqrDistance = Math.pow(x, 2) + Math.pow(y ,2);
		return (sqrDistance < Math.pow(radius, 2));
	}

	public void readConfigFile(String filename) throws IOException {
		FileReader read = new FileReader(filename);
		BufferedReader bufferedReader = new BufferedReader(read);
		String line = "";
		//skip four lines
		line = bufferedReader.readLine();
		if (line.contains("version 3")){
			line = bufferedReader.readLine();
			line = bufferedReader.readLine();
			line = bufferedReader.readLine();
			if (debug) System.out.println(line);
			String [] entries = line.split("\\s+");
			radius = Double.parseDouble(entries[1]);
			line = bufferedReader.readLine();
			entries = line.split("\\s+");
			cylinderminz = Double.parseDouble(entries[1]);
			line = bufferedReader.readLine();
			entries = line.split("\\s+");
			cylindermaxz = Double.parseDouble(entries[1]);
			line = bufferedReader.readLine();
			entries = line.split("\\s+");
			minz = Double.parseDouble(entries[1]);
			line = bufferedReader.readLine();
			entries = line.split("\\s+");
			maxz = Double.parseDouble(entries[1]);
		}
		success = true;
		bufferedReader.close();
	}

	public boolean getSuccess() {
		return success;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/