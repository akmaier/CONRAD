package edu.stanford.rsl.conrad.utils;

import ij.io.FileInfo;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class DicomConfigurationUpdater implements ConfigurationUpdater {

	private Configuration config;
	private String filename = null;
	
	public Configuration getConfiguration() {
		return config;
	}
	
	public void readConfigFromDICOM(String filename) throws IOException{
		String directory = new File(filename).getParent();
		String fileName = new File(filename).getName();
		DicomDecoder dd = new DicomDecoder(directory, fileName);
		dd.setInputStream(new BufferedInputStream(new FileInputStream(directory + "/" + fileName)));;
		FileInfo fi = dd.getFileInfo();
		if (fi.width == 616) fi.width = 620;
		config.getGeometry().setDetectorWidth(fi.width);
		config.getGeometry().setDetectorHeight(fi.height);
		
		dd.setInputStream(new BufferedInputStream(new FileInputStream(directory + "/" + fileName)));;
		
		//System.out.println(dd.getDicomInfo());
		String [] lines = dd.getDicomInfo().split("\n");
		for (String line: lines){
			if (line.contains("Positioner Primary Angle Increment:")){
				double [] angles = parseDICOMFieldForDoubleArray(line);
				config.getGeometry().setPrimaryAngleArray(angles);
				config.getGeometry().setAverageAngularIncrement(DoubleArrayUtil.computeAverageIncrement(angles));
			}
			if (line.contains("Positioner Secondary Angle Increment:")){
				config.getGeometry().setSecondaryAngleArray(parseDICOMFieldForDoubleArray(line));
			}
			if (line.contains("Imager Pixel Spacing:")){
				double [] pixelSpacing = parseDICOMFieldForDoubleArray(line);
				config.getGeometry().setPixelDimensionX(pixelSpacing[0]);
				config.getGeometry().setPixelDimensionY(pixelSpacing[1]);
			}
			if (line.contains("Number of Frames:")){
				config.getGeometry().setProjectionStackSize((int) parseDICOMFieldForDouble(line));
			}
			if (line.contains("Intensifier Size:")){
				config.setIntensifierSize((int) parseDICOMFieldForDouble(line));
			}
			if (line.contains("Distance Source to Patient:")){
				config.getGeometry().setSourceToAxisDistance(parseDICOMFieldForDouble(line));
			}
			if (line.contains("Distance Source to Detector:")){
				config.getGeometry().setSourceToDetectorDistance(parseDICOMFieldForDouble(line));
			}
			if (line.contains("Device Serial Number:")){
				int number = (int) parseDICOMFieldForDouble(line);
				config.setDeviceSerialNumber("" + number);
			}
		}
	}

	private double [] parseDICOMFieldForDoubleArray(String line){
		String [] preprocess = line.split("\\\\");
		String [] temp = preprocess[0].split(" ");
		preprocess[0] = temp[temp.length -1];
		double [] revan = new double [preprocess.length];
		for (int i = 0; i < revan.length; i++){
			revan[i] = Double.parseDouble(preprocess[i]);
		}
		return revan;
	}

	private double parseDICOMFieldForDouble(String line){
		String [] preprocess = line.split(" ");
		return Double.parseDouble(preprocess[preprocess.length - 1]);
	}

	public void readConfiguration() {
		try {
			readConfigFromDICOM(filename);
		} catch (IOException e) {
			System.out.println(e.getLocalizedMessage());
		}
	}

	public void setConfiguration(Configuration config) {
		this.config = config;
	}

	public Configuration getConfig() {
		return config;
	}

	public void setConfig(Configuration config) {
		this.config = config;
	}

	public String getFilename() {
		return filename;
	}

	public void setFilename(String filename) {
		this.filename = filename;
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/