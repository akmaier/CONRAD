import javax.swing.JOptionPane;

import edu.stanford.rsl.apps.gui.roi.EvaluateROI;
import ij.plugin.PlugIn;


public class Apply_ROI_Method implements PlugIn {

	public Apply_ROI_Method(){
		
	}
	
	@Override
	public void run(String arg) {
		EvaluateROI rois [] = EvaluateROI.knownMethods(); 
		EvaluateROI roi = (EvaluateROI) JOptionPane.showInputDialog(null, "Select ROI algorithm: ", "Algorithm selection", JOptionPane.PLAIN_MESSAGE, null, rois, rois[0]);
		try {
			roi.configure();
			roi.evaluate();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/