package edu.stanford.rsl.conrad.geometry.trajectories;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

public class ProjectionTableFileTrajectory extends ConfigFileBasedTrajectory{

	/**
	 * 
	 */
	private static final long serialVersionUID = 8539564781880779305L;
	private int version = 3;
	private boolean success = false;

	public ProjectionTableFileTrajectory(String filename, Trajectory model) throws IOException{
		super(model);
		readConfigFile(filename);
	}

	public ProjectionTableFileTrajectory(Trajectory model) {
		super(model);
	}

	public void readConfigFile(String filename) throws IOException{
		String projectionConfig = filename;
		FileReader read = new FileReader(projectionConfig);
		BufferedReader bufferedReader = new BufferedReader(read);
		String line = "";
		// Read version
		line = bufferedReader.readLine();
		if (line.contains("version 4")) version = 4;
		if (line.contains("version 3")) version = 3;
		if (line.contains("version 5")) version = 5;
		if (line.contains("version 6")) version = 6;
		// skip three lines
		line = bufferedReader.readLine();
		line = bufferedReader.readLine();
		line = bufferedReader.readLine();
		if (version == 4)
			line = bufferedReader.readLine();
		if ((version == 5)||(version == 6)) line = bufferedReader.readLine();
		//read number of matrices
		if (version == 3) {
			numProjectionMatrices = Integer.parseInt(bufferedReader.readLine());
		}
		if ((version == 4)||(version == 5)||(version == 6)){
			String [] elements = bufferedReader.readLine().split("\\s+");
			numProjectionMatrices = Integer.parseInt(elements[0]);
		}
		this.projectionStackSize = numProjectionMatrices;
		projectionMatrices = new Projection[numProjectionMatrices];
		secondaryAngles = new double[numProjectionMatrices];
		primaryAngles = new double[numProjectionMatrices];
		if ((version == 6)){
			line = bufferedReader.readLine();
			line = bufferedReader.readLine();
			line = bufferedReader.readLine();
		}
		
		//skip one line;
		line = bufferedReader.readLine();

		
		
		int count = 0;
		while (line != null){ 
			// projection number
			line = bufferedReader.readLine();
			// angles
			line = bufferedReader.readLine();
			if (line != null){
				String [] elements = line.split(" ");
				this.primaryAngles[count] = Double.parseDouble(elements[0]);
				this.secondaryAngles[count] = Double.parseDouble(elements[1]);
				//read one matrix;
				double [][] projectionMatrix = new double[3][4];
				for (int i =0; i<3;i++){
					line = bufferedReader.readLine();
					elements = line.split(" ");
					for (int j = 0; j < elements.length; j++){
						projectionMatrix[i][j] = Double.parseDouble(elements[j]);
					}

				}
				projectionMatrices[count] = new Projection(new SimpleMatrix(projectionMatrix));
				//skip one line
				count++;
			}
			if (count >= this.numProjectionMatrices) break;
			line = bufferedReader.readLine();
		}
		if (count != numProjectionMatrices){
			success = false;
			bufferedReader.close();
			throw new IOException("Number of Matrices in projection table file does not match the actual number in the file. Please check consistency.");
		} else {
			success = true;
			
			bufferedReader.close();
		}

	}

	@Override
	public boolean getSuccess() {
		return success;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/