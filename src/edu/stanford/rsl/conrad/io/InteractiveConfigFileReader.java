/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import javax.swing.JOptionPane;

public abstract class InteractiveConfigFileReader {
	public static boolean readFileWithFeedback(String message, String initialFilename, ConfigFileParser parser){
		String filename = initialFilename;
		boolean done = false;
		boolean success = false;
		while (!done){
			try {
				parser.readConfigFile(filename);
				done = true;
				success = true;
			} catch (Exception e){
				filename = JOptionPane.showInputDialog(message, filename);
				if (filename == null) {
					//System.out.println(filename);
					done = true;
					success = false;
				}
			}
		}
		return success;
	}
}
