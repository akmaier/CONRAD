package edu.stanford.rsl.conrad.io;

import java.io.IOException;

public class SelectionCancelledException extends IOException {

	private static final long serialVersionUID = 5405797321464193492L;
	
	public SelectionCancelledException(String reason) {
		super(reason);
	}

}
