% Script to create many (e.g. 100) models for Uncertainty and ML

clear all 
close all

folderpath = 'D:\mrst-2017a\modules\vemmech\expfrac\Carmel\ML\';


for i = 51 : 300 % Number of models to create
   
   % Current folder's name 
   curfolder = [folderpath, 'case', num2str(i)];
   
   % Create current folder for the model 
   mkdir(curfolder); 
   
   % Copy and paste base case model to the folder created
   copyfile([folderpath, 'case.m'], curfolder); 
   
   % Rename it
   oldfilename = [[folderpath, 'case'], num2str(i),'\case.m'];  
   newfilename = [folderpath, 'case', num2str(i),'\case', num2str(i), '.m'];
   movefile(oldfilename, newfilename); 
   
      
   % Copy and paste .sh and runmodel.m files to the current folder
   copyfile([folderpath,'simulation.sh'], curfolder); 
   copyfile([folderpath, 'runmodel.m'], curfolder); 
   
   % Open file and append or write ('a') at the end of file
   fileID = fopen([curfolder, '\runmodel.m'], 'a');
   
   str1 = 'run(''case'; 
   str2 = num2str(i); 
   str3 = '.m'');'; 
   command = [str1, str2, str3];   
   
   % Append 'run' command at the end of runmodel.m file and close file
   fprintf(fileID, "%s", command);
   fclose(fileID); 
      
   
   
end