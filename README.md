<h1 align='center'>Neural State-Space Modeling with <br>Latent Causal-Effect Disentanglement<br>
   [<a href='https://miccai2022.org/'>MICCAI-MLMI</a>, <a href=''>arXiv</a>]</h2>
<p>This repository holds the experiments and models as explored in the work, "Neural State-Space Modeling with
Latent Causal-Effect Disentanglement," for the MLMI 2022 Workshop. The work will be published in the upcoming conferece during September 18-22 and this repository is available early for post-acceptance visibility. An arXiv pre-print upload is in the works.</p>

### Overview
Despite substantial progress in deep learning approaches to time-series reconstruction, no existing methods are designed to uncover local activities with minute signal strength due to their negligible contribution to the optimization loss. Such local activities however can signify important abnormal events in physiological systems, such as an extra foci triggering an abnormal propagation of electrical waves in the heart. We discuss a novel technique for reconstructing such local activity that, while small in signal strength, is the cause of subsequent global activities that have larger signal strength. Our central innovation is to approach this by explicitly modeling and disentangling how the latent state of asystem is influenced by <i>potential hidden internal interventions</i>. In a novel neural formulation of state-space models (SSMs), we first introduce <b>causal-effect modeling of the latent dynamics</b> via a <b>system of interacting neural ODEs</b> that separately describes 1) the continuous-time dynamics of the internal intervention, and 2) its effect on the trajectory of the system’s native state. Because the intervention can not be directly observed but have to be disentangled from the observed subsequent effect, we integrate knowledge of the native <i>intervention-free</i> dynamics of a system, and infer the hidden intervention by assuming it to be responsible for differences observed between the actual and hypothetical <i>intervention-free dynamics</i>. We demonstrated a proof of concept of the presented framework on reconstructing ectopic foci disrupting the course of normal cardiac electrical propagation from remote observations.
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/182524083-9407cb3d-e42f-4945-bc9b-f7c3492ab531.png" alt="framework schematic" )/></p>
<p align='center'>Fig 1. Schematic of the prposed ODE-VAE-IM model.</p>

### Citation
If you use portions of this repository or have found use for the model in your research directions, please consider citing:
```
@inproceedings{toloubidokhti2022latentcausalssm,
  title={Neural State-Space Modeling with Latent Causal-Effect Disentanglement},
  author={Toloubidokhti, Maryam and Missel, Ryan and Jiang, Xiajun and Otani, Niels and Wang, Linwei},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  year={2022},
}
```

### Setup
- A requirements.txt file is provided to handle all used packages within the repository, automatically generated using the Python package <a href="https://pypi.org/project/pigar/">pigar</a>. 
- Trained checkpoints for each model are provided in the <code>experiments</code> folder.
- Pre-generated datasets used in the testing reconstruction visualizations are provided in the <code>vals/</code> folder for each model.
- Code on how to build and train neural state-space models (and ODE-VAEs) are provided in <code>Model Code</code> folder.

### Data
We provide data generation scripts in the <code>Data Generation</code> folder. To use the data in this codebase, simply move the
generated files <code>Intervention</code> and <code>Normal</code> to the <code>Pacing</code> and <code>Normal</code> folders respectively. Then toggle the
function parameter <code>newload</code> to True in <code>data_loader.py</code> and run the code to generate each datasets' files.'

The <code>Normal</code> dataset refers to the base dynamics presented in the paper which are native transmembrane potential sets containing 1000 voltage maps, in which the initial excitation locations are chosen randomly across the 100*100mm 2D grid. The <code>Intervention</code> dataset refers to the transmembrane potentials when an extra Foci is present and contains <code>705</code> samples with varying initial locations and times for both the excitation and extra Foci.

<b>Generating the <code>Normal</code> dynamics dataset:</b>
1. (Skip 1 if you wish to use the given forward matrix H.mat) You can use the <code>generate_H_3d.m</code> code to generate the forward matrix with the desired parameters (details are commented in the code). The H matrix used to generate the dataset for this paper, is provided as <code>H.mat</code>. H is loaded in line 20 in the <code>generate_H_3d.m</code>.
2. Use MATLAB to run the <code>normal.m</code> code. The first 1000 iterations generate the TMP-BSP pairs assuming there is only one Pacing location
at the first time-step. The next 1000 generate the extra Foci in a random location in the first time step.
3. Data will be saved in the 'Normal' folder. TMPs will be in 'TMP' folder and corresponding BSPs will be saved in 'BSP' folder.

<b>Generating the <code>Intervention</code> dataset (extra Foci):</b>
1. Use the same H used in the Normal dynamics dataset. H is loaded in line 20 of the code <code>extra_pacing.m</code>.
2. Use MATLAB to run the <code>extra_pacing.m</code> code. After generation of each sample, you are asked whether you want to save this data or not. Input 1 if you wish to save the sample. This option is used to discard the samples were the extra Foci does not happen and the sample
looks exactly like the normal dynamics data.
3. Data will be saved in the 'Intervention' folder. TMPs will be in 'TMP' folder and corresponding BSPs will be saved in 'BSP' folder.

### Models
We provide 3 models in <code>Model Code</code>, utilizing Pytorch-Lightning modules to handle the train/test steps. These include:
- <b>ODE-VAE</b>: The base dynamics model that uses a CNN spatial encoder/decoder to a latent state <code>z<sub>i</sub></code> and a neural ODE dynamics network to perform the forecasting. Given <code>k</code> initial frames of observation, it forecasts fully out to <code>T</code> timesteps.
- <b>ODE-VAE-GRU</b>: The given baseline to compare again, which uses the same ODE dynamics network but additionally includes uses a sliding window of forward observations to update the latent state with a GRU cell at every timestep. The sliding window is fed into the same inital latent state encoder used to initialize <code>z<sub>0</sub></code> 
- <b>ODE-VAE-IM</b>: The proposed intervention model which consists of a system of interactin neural ODEs to separately describe 1.) the latent dynamics of the internal intervention and 2.) what effect it has on the resulting trajectory of the system's native dynamics. It leverages a pre-trained ODE-VAE on the native dynamics and learns another neural ODE F(a) to describe the intervention dynamics. The prediction of the states post-intervention are handled through a coupled ODE system F(z + a), in which the intervention dynamics directly affect the ODE vector predictions in the integration step.

### Running the script
One can either:
- Manually change the hyperparameters in the script and simply run <code>python3 train.py</code>
- Or specify the arguments in the command line, e.g. <code>python3 train.py --model ode_vae --version normal --batch_size 32</code>

### Intervention reconstruction examples

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/182524945-1b5a8f9e-ba01-4fc3-a502-047fe8421a73.png" alt="reconstruction examples" )/></p>
<p align='center'>Fig 2. Reconstruction of electrical propagation in which ectopic foci occurs.</p>

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/182525703-186c4f1e-bf01-4a29-a01b-a7016967a282.png" alt="reconstruction examples 2" )/></p>
<p align='center'>Fig 3. Reconstruction of electrical propagation in which ectopic foci occurs.</p>

### Latent norm ablations

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/182527284-ae2540b5-b841-40d9-8467-3048b31bfc03.png" alt="latent norm examples" )/></p>
<p align='center'>Fig 4. Visualizations of the L2-Norm of system and intervention states over time.</p>
