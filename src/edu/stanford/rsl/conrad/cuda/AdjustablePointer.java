package edu.stanford.rsl.conrad.cuda;

import jcuda.Pointer;

public class AdjustablePointer extends Pointer {
	
	public AdjustablePointer (Pointer pointer, long byteOffset){
		super(pointer, byteOffset);
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/