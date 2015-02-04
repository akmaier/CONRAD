package edu.stanford.rsl.conrad.utils;

import java.io.*;
import javax.swing.filechooser.FileFilter;

/*
String File Filter creates a FileFilter for JFileChooser.
All Files ending with "filt" will be accepted.
*/
public class StringFileFilter extends FileFilter{

	String filter;
	
	public StringFileFilter (String filt){
		filter = filt;
	}
	
	public boolean accept(File file){
		return (file.getAbsolutePath().endsWith(filter)||file.isDirectory());
	}
	
	public String getDescription(){
		return filter;
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/