package edu.stanford.rsl.conrad.geometry.motion;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class MotionUtil {

	public static MotionField get4DSpline() {
		MotionField scene = null;
		String name = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION);
		if (name != null){
			try {
				FileInputStream fis = new FileInputStream(name);
				ObjectInputStream ois = new ObjectInputStream(fis);
				scene = (MotionField) ois.readObject();
				ois.close();
				return scene;
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}
		} else {
			return null;
		}
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/