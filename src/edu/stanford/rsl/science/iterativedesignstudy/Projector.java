package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.numeric.NumericGrid;

public abstract class Projector {


	public abstract NumericGrid project(NumericGrid grid, NumericGrid sino);
	public abstract NumericGrid project(NumericGrid grid, NumericGrid sino, int index);
	
}
