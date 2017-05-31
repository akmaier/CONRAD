package edu.stanford.rsl.tutorial.informatiker;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class MyPhantom extends Grid2D {
	public MyPhantom() {
		super(640, 480);

		// Draw a square with size 100,100 with center 150,150
		for (int x = 100; x <= 200; x++) {
			for (int y = 100; y <= 200; y++) {
				setAtIndex(x, y, 0.5f);
			}
		}

		// Draw a circle at 300,300 with radius 33,333...
		for (int x = 0; x < 640; x++) {
			for (int y = 0; y < 480; y++) {
				if (Math.abs(x - 300) * Math.abs(x - 300) + Math.abs(y - 300) * Math.abs(y - 300) < 1000) {
					setAtIndex(x, y, 1f);
				}
			}
		}

		// Draw a line at the top border
		for (int x = 0; x < 640; x++) {
			setAtIndex(x, 0, 0.75f);
		}
	}
}