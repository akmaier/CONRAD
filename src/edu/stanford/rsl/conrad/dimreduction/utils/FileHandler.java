package edu.stanford.rsl.conrad.dimreduction.utils;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Writer;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class FileHandler {
	private static final String LINE_SEPARATOR = System
			.getProperty("line.separator");


	/**
	 * calls the right method to load the file
	 * 
	 * @param filename
	 *            filename of the file, which will be loaded. with ending .txt
	 *            or .csv
	 * @return the double[][] of the inner-point distances stored in the file
	 * @throws NumberFormatException
	 * @throws IOException
	 * @throws MyException
	 *             if the ending of the file is wrong
	 */
	public static double[][] loadData(String filename)
			throws NumberFormatException, IOException, MyException {
		if (filename.length() == 0) {
			throw new MyException("You have to type in a filename!!");
		} else {
			if (filename.charAt(filename.length() - 1) == 't'
					&& filename.charAt(filename.length() - 2) == 'x'
					&& filename.charAt(filename.length() - 3) == 't') {
				return readTxtFile(filename);
			} else if (filename.charAt(filename.length() - 1) == 'v'
					&& filename.charAt(filename.length() - 2) == 's'
					&& filename.charAt(filename.length() - 3) == 'c') {
				return readCsvFile(filename);
			} else {
				throw new MyException("This is no possible file format!!");
				
			}
		}
	}

	/**
	 * loads the file if the ending is .txt
	 * 
	 * @param filename
	 *            name of the file
	 * @return the double[][] of the inner-point distances stored in the file,
	 *         splitted by tab
	 * @throws NumberFormatException
	 * @throws IOException
	 * @throws MyException
	 */
	public static double[][] readTxtFile(String filename)
			throws NumberFormatException, IOException, MyException {
		String strRead;
		int counterLine = 0;
		int counterRow = 0;
		double[][] distances = null;
		BufferedReader br = null;

		try {
			br = new BufferedReader(new FileReader(filename));
			double max = 0;

			while ((strRead = br.readLine()) != null) {
				if (strRead.charAt(0) == '/' || counterLine == 0) {
					// nothing to do
				} else {
					String line[] = strRead.split("\t");
					if (distances == null) {
						distances = new double[line.length][line.length];
					}

					for (int i = 0; i < line.length; ++i) {
						distances[counterRow][i] = Double.parseDouble(line[i]);
						max = Math.max(max, distances[counterRow][i]);
					}
					++counterRow;
				}
				++counterLine;

			}
			for (int i = 0; i < distances.length; i++) {
				for (int j = 0; j < distances[0].length; j++) {
					distances[i][j] /= max;
				}
			}

		} catch (FileNotFoundException e) {
			throw new MyException("I can not find this file!");
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return distances;
	}

	/**
	 * reads the file if its ending was .csv
	 * 
	 * @param filename
	 *            name of the file
	 * @return the double[][] of the inner-point distances stored in the file, splitted by ","
	 * @throws MyException
	 */
	public static double[][] readCsvFile(String filename) throws MyException {

		double[][] distances = null;
		int counterRow = 0;
		BufferedReader br = null;
		String line = "";

		try {
			double max = 0;

			br = new BufferedReader(new FileReader(filename));

			while ((line = br.readLine()) != null) {

				String[] data = line.split(",");

				if (distances == null) {
					distances = new double[data.length][data.length];
				}
				for (int i = 0; i < data.length; ++i) {
					distances[counterRow][i] = Double.parseDouble(data[i]);
					max = Math.max(max, distances[counterRow][i]);
				}
				++counterRow;

			}
			for (int i = 0; i < distances.length; i++) {
				for (int j = 0; j < distances[0].length; j++) {
					distances[i][j] /= max;
				}
			}
		} catch (FileNotFoundException e) {
			throw new MyException("I can not find this file!");
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		return distances;

	}

	
	
	/**
	 * saves the 3D-coordinates as a .txt file
	 * 
	 * @param x
	 *            x-coordinates
	 * @param y
	 *            y-coordinates
	 * @param z
	 *            z-coordinates
	 * @param name
	 *            filename
	 * @throws IOException
	 */
	public static void save(double[] x, double[] y, double[] z, String name)
			throws IOException {
		Writer out = new BufferedWriter(new FileWriter(name + ".txt"));

		for (int i = 0; i < y.length; ++i) {
			// out.write("test\n");
			if (i > 0 && x[i] != x[i - 1]) {
				out.write(LINE_SEPARATOR);
			}
			out.write(x[i] + " " + y[i] + " " + z[i] + LINE_SEPARATOR);
		}

		out.close();
	}

	/**
	 * saves the 2D-coordinates as a .txt file
	 * 
	 * @param x
	 *            x-coordinates
	 * @param y
	 *            y-coordinates
	 * @param name
	 *            filename
	 * @throws IOException
	 */
	public static void save(double[] x, double[] y, String name)
			throws IOException {
		Writer out = new BufferedWriter(new FileWriter(name + ".txt"));

		for (int i = 0; i < y.length; ++i) {
			out.write(x[i] + " " + y[i] + LINE_SEPARATOR);
		}

		out.close();
	}

	/**
	 * saves the file as a java file
	 * 
	 * @param filename
	 * @param file
	 *            file, that is going to be saved
	 */
	public static void save(String filename, double[] file) {
		OutputStream output = null;
		try {
			output = new FileOutputStream(filename);
			ObjectOutputStream o = new ObjectOutputStream(output);
			o.writeObject(file);
			o.close();
		} catch (IOException e) {
			System.err.println(e);
		} finally {
			try {
				output.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * reads a java file with the filename filename
	 * 
	 * @param filename
	 * @return a double[] of the saved coordinates
	 */
	public static double[] read(String filename) {
		InputStream input = null;

		try {
			input = new FileInputStream(filename);
			ObjectInputStream o = new ObjectInputStream(input);
			double[] savedValues = (double[]) o.readObject();
			return savedValues;
		} catch (IOException e) {
			System.err.println(e);
		} catch (ClassNotFoundException e) {
			System.err.println(e);
		} finally {
			try {
				input.close();
			} catch (Exception e) {
			}
		}
		return null;

	}

	/**
	 * saves a array of pointND in a java file with the name filename
	 * 
	 * @param filename
	 * @param points
	 *            array of pointND that is saved
	 */
	public static void save(String filename, PointND[] points) {
		OutputStream output = null;
		try {
			output = new FileOutputStream(filename);
			ObjectOutputStream o = new ObjectOutputStream(output);
			o.writeObject(points);
			o.close();
		} catch (IOException e) {
			System.err.println(e);
		} finally {
			try {
				output.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * reads the java file with filename filename
	 * 
	 * @param filename
	 * @return a PointND[] of the saved points
	 */
	public static PointND[] readPointND(String filename) {
		InputStream input = null;

		try {
			input = new FileInputStream(filename);
			ObjectInputStream o = new ObjectInputStream(input);
			PointND[] savedPoints = (PointND[]) o.readObject();
			return savedPoints;
		} catch (IOException e) {
			System.err.println(e);
		} catch (ClassNotFoundException e) {
			System.err.println(e);
		} finally {
			try {
				input.close();
			} catch (Exception e) {
			}
		}
		return null;

	}

}