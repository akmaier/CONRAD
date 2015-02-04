package edu.stanford.rsl.conrad.utils;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class FileJoiner {

	String pattern = null;
	String outfile = null;
	String directory = null;
	
	public FileJoiner(String directory, String pattern, String outfile){
		this.pattern = pattern;
		this.outfile = outfile;
		this.directory = directory;
	}
	
	public void joinFiles() throws IOException{
		File dir = new File(directory);
		FileWriter writer = new FileWriter(outfile);
		char [] buffer = new char[1024];
		if (dir.isDirectory()){
			String [] files = dir.list();
			Arrays.sort(files);
			for (int i = 0; i < files.length; i++){
				if (files[i].contains(pattern)) {
					FileReader reader = new FileReader(directory + files[i]);
					int read = 20;
					while ((read = reader.read(buffer)) > 0){
						writer.write(buffer, 0, read);
					}
				}
			}
		}
		writer.close();
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		FileJoiner join = new FileJoiner("D:\\recon\\CLINICAL_CARDIAC_5_3317946_1_1_191_converted\\", "CosineCorr", "D:\\recon\\cosinecorr.float");
		try {
			join.joinFiles();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/