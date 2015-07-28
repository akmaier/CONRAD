package edu.stanford.rsl.conrad.utils;

import java.io.File;

import javax.swing.JFileChooser;

import edu.stanford.rsl.conrad.io.SelectionCancelledException;


public abstract class FileUtil {

	/**
	 * Wrapper for the Swing FileChooser. Prompts the user for a file that matches "filter".
	 * If save is set to "true", the heading will read as "Save as".
	 * If it is false, the heading reads "load".
	 * If the User cancels, an Exception is thrown.
	 * @param filter StringFileFilter for the file type
	 * @param save if true a save dialog is shown
	 * @return absolute path of the file
	 * @throws Exception if the action is cancelled
	 */
	public static String myFileChoose(String filter, boolean save) throws Exception{
		return myFileChoose(null, filter, save);
	}
	
	/**
	 * Wrapper for the Swing FileChooser. Prompts the user for a file that matches "filter".
	 * If save is set to "true", the heading will read as "Save as".
	 * If it is false, the heading reads "load".
	 * If the User cancels, an Exception is thrown.
	 * @param title window title. Will not be set if null.
	 * @param filter StringFileFilter for the file type
	 * @param save if true a save dialog is shown
	 * @return absolute path of the file
	 * @throws Exception if the action is cancelled
	 */
	public static String myFileChoose(String title, String filter, boolean save) throws Exception{
		Configuration.loadConfiguration();
		JFileChooser FC = new JFileChooser();
		if (title != null) FC.setDialogTitle(title);
		if (Configuration.getGlobalConfiguration().getCurrentPath() != null){
			File dir = new File(Configuration.getGlobalConfiguration().getCurrentPath());
			FC.setCurrentDirectory(dir);
		}
		FC.setVisible(true);
		FC.setFileFilter(new StringFileFilter(filter));
		int result = -1;
		if (save) {
			result = FC.showSaveDialog(null);
		} else result = FC.showOpenDialog(null);
		if (result == JFileChooser.CANCEL_OPTION) {
			throw new SelectionCancelledException("Cancelled");
		}
		String filename = FC.getSelectedFile().getAbsolutePath();
		if (save){
			if (! filename.endsWith(filter)){
				filename += filter;
			}
		}
		Configuration.getGlobalConfiguration().setCurrentPath(FC.getCurrentDirectory().getAbsolutePath());
		return filename;
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/