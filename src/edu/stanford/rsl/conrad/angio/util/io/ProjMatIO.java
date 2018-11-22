/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.io;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

public class ProjMatIO {

	public static Projection[] readProjMats(String filename){
		 
		ArrayList<Projection> pMat = null;
		if(filename.contains(".bin")){
			pMat = readBinary(filename);
		}else if( filename.contains(".txt")){
			pMat = readProjTable(filename);
		}else if(filename.contains(".ompl")){
			pMat = readOmpl(filename);
		}else{
			System.err.println("Trajectory file type reader not implemented or extension unknown.");
		}
		Projection[] proj = new Projection[pMat.size()];
		for(int i = 0; i < proj.length; i++){
			proj[i] = pMat.get(i);
		}		
		return proj;
	}
	
	private static ArrayList<Projection> readProjTable(String filename){
		ArrayList<Projection> pMat = new ArrayList<Projection>();
		try{
			FileReader fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);
			String line = br.readLine();
			StringTokenizer tok;
			while(line != null){
				if(line.toLowerCase().contains("format: angle")){
					tok = new StringTokenizer(br.readLine());
					@SuppressWarnings("unused")
					int nMat = Integer.parseInt(tok.nextToken());
				}else if(line.contains("@")){
					SimpleMatrix mat = new SimpleMatrix(3,4);
					br.readLine();// reads primAngles
					for(int k = 0; k < 3; k++){
						tok = new StringTokenizer(br.readLine()); // reads line of matrix
						for(int l = 0; l < 4; l++){
							mat.setElementValue(k, l, Double.parseDouble(tok.nextToken()));
						}
					}
					pMat.add(new Projection(mat));
				}else{
					
				}
				line = br.readLine();
			}
			
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e) {
			return null;
		} catch (IOException e) {
			System.out.println("Reading ProjTable did not work.");
		}
		return pMat;
	}
	
	private static ArrayList<Projection> readBinary(String filename){
		ArrayList<Projection> pMat = new ArrayList<Projection>();
		try {
			if(!filename.contains(".bin")){
				throw new Exception();
			}
			FileInputStream fStream = new FileInputStream(filename);
			// Number of matrices is given as the total size of the file
			// divided by 4 bytes per float, divided by 12 floats per projection matrix
			int nMat = (int) (fStream.getChannel().size() / 4 / (4*3));
			DataInputStream in = new DataInputStream(fStream);
			
			for(int m = 0; m < nMat; m++){
				SimpleMatrix mat = new SimpleMatrix(3,4);
				for(int i = 0; i < mat.getRows(); i++){
					for(int j = 0; j < mat.getCols(); j++){
						byte[] buffer = new byte[4];
					    int bytesRead = in.read(buffer);
					    float val = 0;
					    if(bytesRead == 4){
					    	val = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getFloat();
					    }
					    mat.setElementValue(i, j, val);
					}
				}
				pMat.add(new Projection(mat));
			}
			
			in.close();
			fStream.close();
		} catch (FileNotFoundException e) {
			System.out.println("Trajectory-file not found.");
			return null;
		} catch (Exception e) {
			System.out.println("Rading binary trajectory file did not work.");
		}
		return pMat;
	}
	
	private static ArrayList<Projection> readOmpl(String file){
		ArrayList<Projection> proj = new ArrayList<Projection>();
		try{
			FileReader fr = new FileReader(file);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			while( line != null && !line.isEmpty()){
				StringTokenizer tok = new StringTokenizer(line);
				SimpleMatrix mat = new SimpleMatrix(3,4);
				for(int k = 0; k < 3; k++){
					for(int l = 0; l < 4; l++){
						String t = tok.nextToken();
						if(t.contains("[")){
							t = t.replace("[", "");
						}else if(t.contains("]")){
							t = t.replace("]", "");
						}else if(t.contains(";")){
							t = t.replace(";", "");
						}
						mat.setElementValue(k, l, Double.parseDouble(t));
					}
				}
				proj.add(new Projection(mat));
				line = br.readLine(); // read next line
			}
			
			br.close();
			fr.close();
		} catch (FileNotFoundException e) {
			return null;
		} catch (IOException e) {
			System.out.println("Reading OMPL did not work. Will return null.");
		}
		return proj;
	}
	
	
	
	public static void writeOmpl(Projection[] proj, String file){
		PrintWriter writer;
		try {
			writer = new PrintWriter(file,"UTF-8");
			for(int i = 0; i < proj.length; i++){
				SimpleMatrix p = proj[i].computeP();
				String line = "[";
				for(int k = 0; k < 3; k++){
					for(int l = 0; l < 4; l++){
						line += p.getElement(k, l);
						if(l != 3){
							line += " ";
						}
					}
					if(k != 2){
						line += "; ";
					}else{
						line += "]";
					}
				}
				writer.println(line);
			}
		writer.close();
		
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}		
	}
	
	public static void writeProjTable(Projection[] proj, String file){
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		
		PrintWriter writer;
		try {
			writer = new PrintWriter(file,"UTF-8");
			writer.println("projtable version 3");
			writer.println(dateFormat.format(date));
			writer.println();
			writer.println("# format: angle / entries of projection matrices");
			writer.println(proj.length);
			writer.println();
			for(int i = 0; i < proj.length; i++){
				writer.println("@ "+String.valueOf(i+1));
				SimpleMatrix p = proj[i].computeP();
				double primaryAngle = 0;
				double secondaryAngle = 0;
				writer.println(primaryAngle+" "+secondaryAngle);
				for(int k = 0; k < 3; k++){
					String line = "";
					for(int l = 0; l < 4; l++){
						line += p.getElement(k, l) + " ";
					}
					writer.println(line);
				}
				writer.println();
			}
		writer.close();
		
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}		
	}
	
}
