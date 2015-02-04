/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.Serializable;

public interface SafeSerializable extends Serializable {

	/**
	 * Sets all data objects in the serialized object to null which do not implement the Serializable interface. It can also be used to save space in the serialized form.
	 * Configuration parameters are preserved. Processing data is discarded.
	 */
	public void prepareForSerialization();

}
