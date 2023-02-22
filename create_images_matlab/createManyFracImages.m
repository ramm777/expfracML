% Create images of 247 initial fractures images upon the first contact(Gfracs)
% based on the simulation results

clear all
close all


whatToCreate = 'Gfrac_images'; % 'Gfrac_maxwidth', 'Gfrac_images'
datapath = 'D:\mrst-2017a\modules\vemmech\RESULTS\Synthetic2\LMd_case5-1full';
casesomit = [15,16,17,19,33]; % [198, 199, 200]; 
subcasesnum = 37; 
max_width_selected = 10e-3; % calculated in 247 => 7.000109e-05, or take 10e-3


switch whatToCreate

    case('Gfrac_images')

    figure(1); 
    for i = 1:subcasesnum
        
        fprintf('i: %d\n', i);
        
        if any(i == casesomit)
            fprintf('Omitting subcase: %d\n', i);
            continue
        end
        
        
        load([datapath, '\case5_', num2str(i), '\case5_', num2str(i)], 'Ginit');
        load([datapath, '\case5_', num2str(i), '\case5_', num2str(i)], 'fraccells');
        Gfrac = extractSubgrid(Ginit, fraccells);

        
        plotGrid(Gfrac, 'EdgeAlpha', 0, 'FaceColor', 'white'); % 
        ss = get(groot, 'Screensize');
        set(1, 'Position', [ss(1)*1900, ss(2)-150, 900, 730]);
        axis equal

        min_xcoord = min(Gfrac.nodes.coords(:, 1));
        max_xcoord = min_xcoord + max_width_selected;  
        xlim([min_xcoord - max_width_selected/100, max_xcoord- max_width_selected/100]); ylim([0, 10e-3])
      
        
        w = 5*(max_xcoord - min_xcoord);
        h = w; % 5*(6.7e-4); 
        set(gca,'Position', get(gca,'OuterPosition'));
        set(gcf,'inverthardcopy','off'); 
        set(gca,'Color','k');
        set(gcf,'Color','k');
        set(gca,'visible','off'); 
        set(gcf,'PaperUnits','inches','PaperPosition',[0 0 39.3701*w 39.3701*h])

        filename = [num2str(i), '.jpg'];
        print(filename, '-djpeg', '-r5000');

        clf(1);



    end

    
    case('Gfrac_maxwidth')
        
        allmechapers = [];
        max_width_vector = nan(1, subcasesnum);
    
        for i = 1 : subcasesnum 
            
            fprintf('i: %d\n', i);
            
            if any(i == casesomit)
                fprintf('Omitting subcase: %d\n', i);
                continue
            end
            
            load([datapath, '\case5_', num2str(i), '\case5_', num2str(i)], 'Ginit');
            load([datapath, '\case5_', num2str(i), '\case5_', num2str(i)], 'fnodes');
                        
                        
            % Find max_width of Gfrac
            [allmechapers, ~] = mechapers(Ginit, fnodes, 1, allmechapers, [], 'mode', 'distance', 'outputSameSize', 0);
            max_width = max(allmechapers);     
            max_width_vector(i) = max_width; 
            
        end
        
                    
        max_width_all = max(max_width_vector); 
        fprintf('max width all: %d\n', max_width_all)

        
        
end
