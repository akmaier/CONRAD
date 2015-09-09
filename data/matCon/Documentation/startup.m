% Set which name of the CONRAD directory, usually CONRAD
CONRADPACKAGE = 'CONRAD';
% Set the path where CONRAD is located
GLOBALCONRADPATH = 'C:\Users\berger\Documents\Development';

disp('Adding CONRAD Matlab methods to the Matlab path')
addpath(genpath([GLOBALCONRADPATH filesep CONRADPACKAGE  filesep 'data' filesep 'matCon']),'-end');

disp('Adding CONRAD Source to the Matlab path')
addpath(genpath([GLOBALCONRADPATH filesep CONRADPACKAGE]),'-end');

import java.*
c=CONRAD(GLOBALCONRADPATH);
c.initialize;