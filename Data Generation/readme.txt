Generating the Noraml dynamics dataset:
1- (Skip 1 if you wish to use the given forward matrix H.mat) You can use the generate_H_3d.m code to generate the forward matrix with the desired parameters (details are commented in the code). The 
H matrix used to generate the dataset for this paper, is provided as H.mat. H is loaded in line 20 in the generate_H_3d.
2- Use MATLAB to run the normal.m code. The first 1000 iterations generate the TMP-BSP pairs assuming there is only one Pacing location
at the first time-step. The next 1000, generate the extra Foci in a random location in the first time step.
3- Data will be saved in the 'Normal' folder. TMPs will be in 'TMP' folder and corresponding BSPs will be saved in 'BSP' folder.

Generating the intervention dataset (extra Foci):
1- Use the same H used in the Normal dynamics dataset. H is loaded in line 20 of the code extra_pacing.m.
2- Use MATLAB to run the extra_pacing.m code. After generation of each sample, you are asked weather you want to save this data or not. 
Input 1 if you wish to save the sample. This option is used to discard the samples were the extra Foci does not happen and the sample
looks exactly like the normal dynamics data.
3- Data will be saved in the 'Intervention' folder. TMPs will be in 'TMP' folder and corresponding BSPs will be saved in 'BSP' folder.