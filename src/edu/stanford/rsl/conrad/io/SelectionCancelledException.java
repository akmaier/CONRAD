/*
 * Copyright (C) 2015 Benedikt Lorch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.io;

import java.io.IOException;

public class SelectionCancelledException extends IOException {

	private static final long serialVersionUID = 5405797321464193492L;
	
	public SelectionCancelledException(String reason) {
		super(reason);
	}

}
