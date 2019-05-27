package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.numeric.NumericGrid;

public abstract interface Backprojector {
	public abstract NumericGrid backproject(NumericGrid sino, NumericGrid grid);
	public abstract NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) ;
}
