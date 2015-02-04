/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/**
 * DISCLAIMER: This reader only reads the points. No Triangle information is
 * read in
 * 
 * @author Marco Boegel
 * 
 */
public class VTKMeshReader {

	/**
	 * The read-in points are saved here
	 */
	public static ArrayList<PointND> pts = new ArrayList<PointND>();

	/**
	 * Getter for the read-in meshpoints
	 * 
	 * @return read-in meshpoints
	 */
	public ArrayList<PointND> getPts() {
		return pts;
	}

	/**
	 * Sets a list of points
	 * 
	 * @param pts
	 */
	public void setPts(ArrayList<PointND> pts) {
		VTKMeshReader.pts = pts;
	}

	/**
	 * Reads the point information from a .vtk file
	 * 
	 * @param filename
	 *            Name of the file
	 * @throws IOException
	 */
	public void readFile(String filename) throws IOException {

		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);

		// read and discard header information
		br.readLine();
		br.readLine();
		br.readLine();
		br.readLine();

		String line = br.readLine();
		StringTokenizer tok = new StringTokenizer(line);
		String t = tok.nextToken();
		t = tok.nextToken();
		int numPoints = Integer.parseInt(t);

		// read points
		for (int i = 0; i < numPoints;) {
			line = br.readLine();
			tok = new StringTokenizer(line);
			int nrPts = tok.countTokens();
			nrPts /= 3;
			for (int j = 0; j < nrPts; j++) {
				PointND p = new PointND(Float.parseFloat(tok.nextToken()),
						Float.parseFloat(tok.nextToken()), Float.parseFloat(tok
								.nextToken()));
				pts.add(p);
			}

			i += nrPts;
		}

		fr.close();
		br.close();

	}

}
