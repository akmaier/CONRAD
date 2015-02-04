package edu.stanford.rsl.conrad.utils;

import java.beans.XMLDecoder;
import java.beans.XMLEncoder;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

public class XmlUtils {
	public static boolean serializeObject(File file, Object o) {
		if (file.exists()) {
			System.out
					.println("Warning! Object Already Exists \n Replacing File.");
		}
		try {
			FileOutputStream os = new FileOutputStream(file);
			XMLEncoder encoder = new XMLEncoder(os);
			encoder.writeObject(o);
			encoder.close();
		} catch (Exception f) {
			System.out.println("Warning! Write was Unsuccessful");
			return false;
		}
		return true;
	}

	public static Object deserializeObject(File file) {
		Object data = null;
		try {
			FileInputStream os = new FileInputStream(file);
			XMLDecoder decoder = new XMLDecoder(os);
			data = decoder.readObject();
			decoder.close();
		} catch (FileNotFoundException f) {
			data = null;
			f.printStackTrace();
		}
		return data;
	}
	
	public static void exportToXML(Object toExport, String filename) throws Exception{
		Thread.currentThread().setContextClassLoader(Configuration.class.getClassLoader());
		XMLEncoder oos = new XMLEncoder (new FileOutputStream(filename));
		oos.writeObject(toExport);
		oos.close();
	}
	
	public static void exportToXML(Object toExport) throws Exception{
		String filename = FileUtil.myFileChoose(".xml", true);
		exportToXML(toExport,filename);
	}

	public static Object importFromXML() throws Exception{
		String filename = FileUtil.myFileChoose(".xml", false);
		return importFromXML(filename);
	}
	
	public static Object importFromXML(String filename) throws Exception{
		Thread.currentThread().setContextClassLoader(Configuration.class.getClassLoader());
		XMLDecoder ois = new XMLDecoder (new FileInputStream(filename));
		Object toImport = ois.readObject();
		ois.close();
		return toImport;
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/