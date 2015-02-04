package edu.stanford.rsl.conrad.utils;

import javax.swing.JTextField;

import edu.stanford.rsl.apps.gui.FileTextFieldTransferHandler;

public abstract class GUIUtil {
	public static  void enableDragAndDrop(JTextField field){
		field.setDragEnabled(true);
		FileTextFieldTransferHandler th2 = new FileTextFieldTransferHandler(field);
		field.setTransferHandler(th2);
	}
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/