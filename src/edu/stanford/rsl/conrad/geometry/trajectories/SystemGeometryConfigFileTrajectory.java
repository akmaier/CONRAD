package edu.stanford.rsl.conrad.geometry.trajectories;

//TODO: Use our own matrices instead of Jama.Matrix

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.io.ConfigFileParser;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ConfigurationUpdater;



public class SystemGeometryConfigFileTrajectory extends ConfigFileBasedTrajectory implements ConfigurationUpdater, ConfigFileParser {

	private static final long serialVersionUID = 2773005628961913345L;
	
	private double [] volumeSize = null;
	private double [] volumeResolution = null;
	private double [] detectorResolution = null;
	private double [] detectorPixels = null;
	private double averageAngularIncrement = -1;
	private int numProjections = -1;
	private SimpleMatrix transformToNextProjection;
	private double SID = -1;
	private double SAD = -1;
	private boolean success = false;
	
	private Configuration config = null;
	
	
	public SystemGeometryConfigFileTrajectory(String filename, Trajectory model) throws IOException{
		super(model);
		readProjectionMatrices(filename);
	}
	
	public SystemGeometryConfigFileTrajectory(Trajectory model) {
		super(model);
	}

	private double [] readDoubleArrayLine(String line){
		String [] values = line.split("\\s+");
		ArrayList<String> list = new ArrayList<String>();
		for(int i = 0; i < values.length; i++){
			if (! values[i].equals("")) list.add(values[i]);
		}
		double [] revan = new double[list.size()];
		for(int i = 0; i < list.size(); i++){
			revan[i] = Double.parseDouble(list.get(i));
		}
		return revan;
	}
	
	private Projection newProjection (double [][] matrix){
		return new Projection(new SimpleMatrix(matrix));
	}

	private void readProjectionMatrices(String projectionConfig) throws IOException{
		FileReader read = new FileReader(projectionConfig);
		BufferedReader bufferedReader = new BufferedReader(read);
		String line = "";
		//skip first comment line;
		line = bufferedReader.readLine();
		// readVolumeInformation
		volumeResolution = readDoubleArrayLine(bufferedReader.readLine());
		volumeSize = readDoubleArrayLine(bufferedReader.readLine());
		// Comment line
		line = bufferedReader.readLine();
		// Detector info
		double [] detectorx = readDoubleArrayLine(bufferedReader.readLine());
		double [] detectory = readDoubleArrayLine(bufferedReader.readLine());
		detectorResolution = new double [2];
		detectorResolution[0] = detectorx[0];
		pixelDimensionX = detectorx[0];
		detectorResolution[1] = detectory[1];
		detectorPixels = readDoubleArrayLine(bufferedReader.readLine());
		// Comment line
		line = bufferedReader.readLine();
		//read number of matrices
		numProjectionMatrices = Integer.parseInt(bufferedReader.readLine());
		projectionMatrices = new Projection[numProjectionMatrices];
		int count = 0;
		while (count < numProjectionMatrices){ 
			//read one matrix;
			double [][] projectionMatrix = readMatrix(bufferedReader, 4, 3);
			projectionMatrices[count++] = newProjection(projectionMatrix);
		}
		if (count != numProjectionMatrices){
			throw new IOException("Number of Matrices in projection table file does not match the actual number in the file. Please check consistency.");
		}
		//read number of projections
		numProjections = Integer.parseInt(bufferedReader.readLine());
		transformToNextProjection = new SimpleMatrix(readMatrix(bufferedReader, 4, 4));
		if (numProjectionMatrices < numProjections){
			Projection [] newMatrices = new Projection[numProjections];
			newMatrices[0] = projectionMatrices[0];
			for (int i = 1; i < numProjections; i++){
				newMatrices[i] = new Projection(SimpleOperators.multiplyMatrixProd(newMatrices[i-1].computeP(), transformToNextProjection));
			}
			projectionMatrices = newMatrices;
			numProjectionMatrices = numProjections;
		}
		double [] [] scaleProjectionCoordinates = new double [3][3];
		scaleProjectionCoordinates[0][0] = 1.0 / detectorResolution[0];
		scaleProjectionCoordinates[1][1] = 1.0 / detectorResolution[1];
		scaleProjectionCoordinates[2][2] = 1.0;
		Jama.Matrix scale = new Jama.Matrix(scaleProjectionCoordinates);
		for (int i = 0; i < projectionMatrices.length; i++){
			projectionMatrices[i] = new Projection(SimpleOperators.multiplyMatrixProd(projectionMatrices[i].computeP().transposed(), new SimpleMatrix(scale)).transposed());
		}
		while ((line = bufferedReader.readLine()) != null){
			String [] elements = line.split("\\s+");
			if (line.contains("SID")){
				String target = elements[3];
				target = target.substring(0, target.length()-2);
				SID = Double.parseDouble(target);
			}
			if (line.contains("Rotation angle per view")){
				String target = elements[6];
				target = target.substring(0, target.length());
				averageAngularIncrement = Double.parseDouble(target);
			}
			if (line.contains("SAD")){
				String target = elements[3];
				target = target.substring(0, target.length()-2);
				SAD = Double.parseDouble(target);
			}
		}
		if (averageAngularIncrement != -1){
			primaryAngles = new double [projectionMatrices.length];
			double current =0;
			for (int i = 0; i < primaryAngles.length; i++){
				primaryAngles[i] = current;
				current += averageAngularIncrement;
			}
			System.out.println(current + " " + averageAngularIncrement);
		}
		
		success = true;
		bufferedReader.close();
	}

	private double [][] readMatrix(BufferedReader bufferedReader, int width, int height) throws IOException{
		double [][] projectionMatrix = new double[height][width];
		for (int i =0; i<height;i++){
			String line = bufferedReader.readLine();
			projectionMatrix[i] = readDoubleArrayLine(line);
		}
		return projectionMatrix;
	}
	
	@Override
	public Configuration getConfiguration() {
		return config;
	}

	@Override
	public void readConfiguration() {
		super.setReconDimensions(volumeSize);
		super.setReconVoxelSizes(volumeResolution);
		super.setDetectorWidth((int) detectorPixels[0]);
		super.setDetectorHeight((int) detectorPixels[1]);
		super.setPixelDimensionX(detectorResolution[0]);
		super.setPixelDimensionY(detectorResolution[1]);
		super.setProjectionStackSize(numProjections);
		super.setSourceToAxisDistance(SAD);
		super.setSourceToDetectorDistance(SID);
		super.setPrimaryAngleArray(primaryAngles);
		super.setAverageAngularIncrement(averageAngularIncrement);
		config.setVolumeOfInterestFileName(null);
		//System.out.println("Projections " + super.getProjectionStackSize());
	}

	@Override
	public void setConfiguration(Configuration config) {
		this.config = config;
	}

	@Override
	public void readConfigFile(String filename) throws IOException {
		readProjectionMatrices(filename);
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