package edu.stanford.rsl.apps.gui;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class TimeLinePlot {

	private static String dataFileString = "E:\\bmw\\bmw2.csv";
	private static String outDir = "E:\\bmw\\";
	private static int vinLoc = 0;
	private static int dateLoc = 1;
	private static int detailsLoc = 2;
	
	
	public static void main (String [] args){
		new ImageJ();

		// read data file
		ArrayList<String []> entries = readFile(dataFileString);
		long [] times = new long[entries.size()];

		// compute data as long
		DateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
		for (int i = 0; i < entries.size(); i++){
			System.out.println(entries.get(i)[dateLoc]);
			try {
				times[i] = formatter.parse(entries.get(i)[dateLoc]).getTime();
			} catch (ParseException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}	
		}
		String currentVin = "";
		for (int i = 0; i < entries.size(); i++){
			if (!entries.get(i)[vinLoc].equals(currentVin)){
				currentVin =entries.get(i)[vinLoc];
				FloatProcessor fl = drawImage(entries, times, currentVin, i);
				ImagePlus imp = new ImagePlus();
				ImageStack stack = new ImageStack(fl.getWidth(),fl.getHeight());
				stack.addSlice(currentVin, fl);
				imp.setStack(currentVin, stack);
				IJ.save(imp, outDir+currentVin +".jpg");
			}
		}
		


	}

	public static FloatProcessor drawImage(ArrayList<String []>entries, long [] times, String currentVin, int startIndex){
		int yPos = 50;
		int xPos = 30;
		int textColor = 50;
		int lineColor = 125;

		long max = -Long.MAX_VALUE;
		long min = Long.MAX_VALUE;
		for (int i = 0; i < entries.size(); i++){
			if (entries.get(i)[vinLoc].equals(currentVin)) {
				if (times[i]>max) max = times[i];
				if (times[i]<min) min = times[i];
			}
		}

		// draw image
		FloatProcessor fl = new FloatProcessor(840, 480);
		fl.setColor(150);
		fl.fill();
		fl.setColor(lineColor);
		fl.drawLine(xPos, yPos+40, xPos+440, yPos+40);
		fl.drawLine(xPos, yPos, xPos, yPos +40);
		fl.drawLine(xPos+440, yPos, xPos+440, yPos+80);
		fl.setColor(textColor);
		fl.drawString("Series of Events", xPos-20, yPos);

		//draw lines
		for (int i = 0; i < entries.size(); i++){
			if (entries.get(i)[vinLoc].equals(currentVin)) {
				int xloc = (int) (440 * (((double)times[i]-min)/(max-min)));
				int yloc = (25*(i-startIndex))+60;
				fl.setColor(lineColor);
				fl.drawLine(xPos+xloc, yPos+40, xPos+xloc, yPos + yloc);
				fl.drawLine(xPos+xloc, yPos + yloc, xPos+xloc+5, yPos + yloc);
			}
		}
		// draw text
		for (int i = 0; i < entries.size(); i++){
			if (entries.get(i)[vinLoc].equals(currentVin)) {
				int xloc = (int) (440 * (((double)times[i]-min)/(max-min)));
				int yloc = (25*(i-startIndex))+60;
				fl.setColor(textColor);
				fl.drawString(entries.get(i)[dateLoc] + ": " + entries.get(i)[detailsLoc], xPos+xloc+10, yPos+yloc +10);
			}
		}
		return fl;
	}

	public static ArrayList<String[]> readFile (String filename){
		ArrayList<String[]> entries = new ArrayList<String[]>();
		try {
			BufferedReader bf = new BufferedReader(new FileReader(filename));
			String line = bf.readLine();
			while(line != null) {
				line = bf.readLine();
				if (line!=null) entries.add(line.split("\\t"));
			}
			bf.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return entries;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
