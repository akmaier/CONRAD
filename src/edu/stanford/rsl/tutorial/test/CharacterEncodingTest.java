/*
 * Copyright (C) 2015 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.test;

/**
 * This is a test case to check whether the character encoding of your eclipse project is correctly configured.
 * If you get an error compiling this file, you need to set your character encoding to "UTF8".
 * To do this right click the top source folder "src" and select properties. In the "resource" tab, in text file encoding select
 * "other" and set it to "UTF-8". This value will then be inherited to all source sub-folders.
 * 
 * @author akmaier
 *
 */
public class CharacterEncodingTest {

	public static void main(String[] args) {
		int umlautü = 10;
		System.out.println("UTF8 correctly configured: " + umlautü);
	}

}
