/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import edu.stanford.rsl.conrad.data.generic.GenericGridOperatorInterface;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;

public interface ComplexGridOperatorInterface extends GenericGridOperatorInterface<Complex> {
	public void conj(final ComplexGrid grid);
	public void abs(final ComplexGrid grid);
}
