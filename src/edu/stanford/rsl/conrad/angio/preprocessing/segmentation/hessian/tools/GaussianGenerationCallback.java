/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.tools;

public interface GaussianGenerationCallback {

    /* Any proportion >= 1.0 indicates completion.  A proportion less
     * than zero indicates that the generation of the Gaussian has
     * been cancelled. */

    public void proportionDone( double proportion );

}