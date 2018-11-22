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

public class EcgIO {

	public static double[] readEcg(String filename){
		ArrayList<Double> e = new ArrayList<Double>();
		FileReader fr;
		try {
			fr = new FileReader(filename);		
			BufferedReader br = new BufferedReader(fr);			
			String line = br.readLine();
			while(line != null){
				StringTokenizer tok = new StringTokenizer(line);
				e.add(Double.parseDouble(tok.nextToken()));
				line = br.readLine();
			}
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		double[] ecg = new double[e.size()];
		for(int i = 0; i < e.size(); i++){
			ecg[i] = e.get(i);
		}
		return ecg;
	}
	
	public static void writeEcg(String filename, double[] ecg){
		File f = new File(filename);
		f.mkdirs();
		
		PrintWriter writer;
		try {
			writer = new PrintWriter(filename,"UTF-8");
			for(int i = 0; i < ecg.length; i++){
				writer.println(String.valueOf(ecg[i]));
			}
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}	
	}
}
