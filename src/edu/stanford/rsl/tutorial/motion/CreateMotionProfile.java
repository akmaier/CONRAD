package edu.stanford.rsl.tutorial.motion;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.XmlUtils;

public class CreateMotionProfile {
	public static void main(String[] args){
		Configuration.loadConfiguration();	
		int nrProj = Configuration.getGlobalConfiguration().getGeometry().getNumProjectionMatrices();
		ArrayList<double[][][]> motionList= new ArrayList<double[][][]>();
		motionList.add(new double[nrProj][][]);
		motionList.add(new double[nrProj][][]);
		double maxz = 2.5;
		double maxxy = 5;
		
		for (int i = 0; i < nrProj; i++) {
			// Rotation 
			motionList.get(0)[i] = SimpleMatrix.I_3.copyAsDoubleArray();
			// translation part:
			motionList.get(1)[i] = new double[3][1];
			motionList.get(1)[i][2][0] = ((double)(i)/nrProj) * maxz;
			motionList.get(1)[i][0][0] = 2.5 - (i%2) * maxxy;
			motionList.get(1)[i][1][0] = 2.5 - (i%2) * maxxy;
		}
		try {
			XmlUtils.exportToXML(motionList, "/Users/maier/Documents/data/ERC Grant/motionpattern_extrem.xml");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
