%%
%
% Class to initialize the java software CONRAD for its use in Matlab
%
%
%%
classdef CONRAD < hgsetget
    
    properties
        isInitialized = 0;
        ConradPath = '';
    end
    
    
    methods
        function obj = CONRAD(path)
            obj.ConradPath=path;
        end
        
        function obj = set.isInitialized(obj,value) % Handle class
            obj.isInitialized = value;
        end
        function value = get.isInitialized(obj) % Handle class
            value = obj.isInitialized;
        end
        
        function obj = set.ConradPath(obj,path)
            if (nargin < 2 || ~isdir(path))
                obj.ConradPath = uigetdir;
            else
                obj.ConradPath = path;
            end
        end
        
        function val = get.ConradPath(obj)
            val = obj.ConradPath;
        end
        
        function obj = initialize(obj)
            if (obj.isInitialized)
                clear java;
            end
            p = cd;
            lib = [obj.ConradPath filesep 'CONRAD' filesep 'lib'];
            cd(lib);
            list = cellstr(ls);
            if ispc
                list = list(3:end);
            elseif isunix
                list = textscan(list{:},'%s');
                list = list{:};
            else
                list = cellstr(ls);
                list = list(3:end);
            end
            for i=1:size(list,1)
                if (isempty([strfind(list{i},'gluegen'), strfind(list{i},'jocl'), strfind(list{i},'joal'), strfind(list{i},'jogl')]) && ~isempty(strfind(list{i},'.jar')))
                    javaaddpath([lib filesep list{i}]);
                end
            end
            cd(p);
            javaaddpath([obj.ConradPath filesep 'CONRAD' filesep 'src']);
            addpath([obj.ConradPath filesep 'CONRAD' filesep 'src']);
            javaaddpath([obj.ConradPath filesep 'CONRAD']);
            addpath([obj.ConradPath filesep 'CONRAD']);
            obj.isInitialized = 1;
        end
        
        
        function RecoPipeline(obj)
            if (~obj.isInitialized)
                obj.initialize;
            end
            import edu.stanford.rsl.*
            conrad.utils.CONRAD.setup();
            a=apps.gui.ReconstructionPipelineFrame();
            a.setVisible(1);
        end
        
        
    end
    
end