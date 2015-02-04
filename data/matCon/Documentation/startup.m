disp('Adding CONRAD Matlab methods to the Matlab path')
addpath(genpath('MatCon'),'-end');

disp('Adding CONRAD_Source files to the Matlab path')
addpath(genpath('D:\Development\CONRAD\CONRAD'),'-end');

import java.*
c=CONRAD('D:\Development\CONRAD');
c.initialize;