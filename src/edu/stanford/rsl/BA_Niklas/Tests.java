package edu.stanford.rsl.BA_Niklas;

import java.io.IOException;
import java.util.Random;


import org.apache.commons.math3.distribution.NormalDistribution;
import org.math.plot.utils.Array;

public class Tests {
	public static void main(String args[]) 
    { 
		
		String command = "py C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/PythonScripts/test.py";
		try {
			Process process = Runtime.getRuntime().exec(command);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

}
}