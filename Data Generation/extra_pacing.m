clear;
close all;

Nx = 100; Ny = 100; % 2D Nx-by-Ny systemm of nodes.
Dx = 0.5; Dy = 0.5; % node spacing in the x and y directions.

% Gap junction 2D resistivity in the x-direction (rho_x*Dx/Dy = resistance):
rho_x = 1.0;
% Gap junction 2D resistivity in the y-direction (rho_x*Dy/Dx = resistance):
rho_y = 1.0;

% Inline function that calculates the node no. corresponding
% to the node located on the grid at (ix,iy):
node_no = @(ix,iy) ix + Nx*(iy-1);


tmp_path = 'Pacing/Intervention/TMP/';
bsp_path = 'Pacing/Intervention/BSP/';
% get H
H = load('H.mat')
H = H.H;

rng(1,'twister');


a = 2;
b = 99;


% random coordinates of the initial pacing
start_xs = fix((b-a).*rand(10000,1));
start_ys = fix((b-a).*rand(10000,1));

a2 = 2;
b2 = 99;


% random coordinates of the second pacing (intervention)
pace_xs = fix((b2-a2).*rand(10000,1));
pace_ys = fix((b2-a2).*rand(10000,1));


% random time steps for the intervention to happen
t1 = 200;
t2 = 700;
start_pace_time = fix((t2-t1).*rand(10000,1)) + t1;



save_sample = 1;

% signal to noise ratio for the BSP datas
snr = 30;

for i=1:10000
    
    start_x = start_xs(i);
    start_y = start_ys(i);
    pace_x = pace_xs(i);
    pace_y = pace_ys(i);

    % Matrix representing gap junction connectivity.
    % One row and one column for each of the nodes in the system.
    A = sparse(Nx*Ny,Nx*Ny);

    

    % Add resistors oriented in the horizontal direction:
    for ix = 1:(Nx-1)
        for iy = 1:Ny
            % Connect a resistor of value r*Dx between
            % nodes (ix,iy) and (ix+1,iy):
            i1 = node_no(ix,iy); % node number corresponding to (ix,i).
            i2 = node_no(ix+1,iy); % node number corresponding to to node (ix+1,iy)
            A(i1,i1) = A(i1,i1) + 1/(rho_x*Dx/Dy);
            A(i1,i2) = A(i1,i2) - 1/(rho_x*Dx/Dy);
            A(i2,i1) = A(i2,i1) - 1/(rho_x*Dx/Dy);
            A(i2,i2) = A(i2,i2) + 1/(rho_x*Dx/Dy);
        end
    end

    % Add resistors oriented in the vertical direction:
    for iy = 1:(Ny-1)
        for ix = 1:Nx
            % Connect a resistor of value r*Dx between
            % nodes (ix,iy) and (ix+1,iy):

            i1 = node_no(ix,iy); % node number corresponding to (ix,i).
            i2 = node_no(ix,iy+1); % node number corresponding to to node (ix,iy+1)
            A(i1,i1) = A(i1,i1) + 1/(rho_y*Dy/Dx);
            A(i1,i2) = A(i1,i2) - 1/(rho_y*Dy/Dx);
            A(i2,i1) = A(i2,i1) - 1/(rho_y*Dy/Dx);
            A(i2,i2) = A(i2,i2) + 1/(rho_y*Dy/Dx);

        end
    end



    % Initialize nodal voltages (i.e, the membrane potentials).
    % Also initialize other dynamical variables
    % Voltages will be stored in the Nx*Ny-by-1 array V.
    % For the Fitzhugh-Nagumo model, the other variable is W.
    V = zeros(Nx*Ny,1); W = zeros(Nx*Ny,1);
    for ix = 1:Nx
        for iy = 1:Ny
            if ( sqrt((ix-start_x)^2+(iy-start_y)^2) <= 2 ); V(node_no(ix,iy)) = +1.0;
            else; V(node_no(ix,iy)) = -1.2;
            end
            W(node_no(ix,iy)) = -0.62;
        end
    end


    % Run the simulation (Fitzhugh-Nagum  o model; forward Euler method):
    epsilon = 0.23; gamma = 0.72; beta = 0.8; % (Fitzhugh-Nagumo parameters);
    c = 1; % membrane capacitance per unit area
    Dt = 0.025; % timestep size
    Nt = 1400;
    
    heart_potential = zeros(Nx*Ny, Nt);

    for it = 1:Nt
        % Display current timestep no. every so often:
        if (mod(it,200)==0); fprintf('Timestep no. %i\n',it); end
        % Nonlinear nodal ion channel current per unit area:
        i_NL = - (1/epsilon)*(V - V.*V.*V/3 - W);
        V = V + (Dt/(c*Dx*Dy)) * ( -A*V - i_NL*Dx*Dy); % Advance V to the next timestep.
        W = W + Dt * epsilon * ( V - gamma*W + beta);
        heart_potential(:,it) =  V;
    %     Plot V vs. x and y every so often:

        if (mod(it,50)==0)
            Vplot = zeros(Nx,Ny);
            Vplot(:) = V; % Vplot is V reformatted as a 2D array
            pcolor(Vplot'); caxis([-2 2]); shading interp;
    %         title(sprintf('Sim for rand seed %i: Time = %f\n',s,it*Dt));
            xlabel('x'); ylabel('y');
            colorbar; drawnow;

        end
 
        
%         create the extra foci dynamics
        if (it == start_pace_time(i))
            for ix = 1:Nx
                for iy = 1:Ny
                    if ( sqrt((ix-pace_x)^2+(iy-pace_y)^2) <= 2 );
                        V(node_no(ix,iy)) = +1.0;
                    end
                end
            end
        end
         
    tmp = heart_potential(:,1:50:end);
    
%     do you wish to save this sample?
       
    end
  
    prompt = 'save? ';
    x = input(prompt)
    if x == 1
        time = size(tmp, 2);
        dim_heart = sqrt(size(tmp, 1));
        mea = H * tmp;
        
%         add noise to body surface potentials
        bsp_ = gen_noise(mea, snr);
        dim = sqrt(size(bsp_, 1));
        bsp_ = bsp_';
        tmp = tmp';
        %     pause;
        %     reshape ansd save the results
        tmp = reshape(tmp, time, dim_heart, dim_heart);
        bsp = reshape(bsp_, time, dim, dim);
        
        
        tmp_name = strcat('pace_tmp_', 'pace1_' ,string(start_x), '_', string(start_y), '_pace2_' ,string(pace_x), '_', string(pace_y), '_time_', string(start_pace_time(i)), '.mat');
        bsp_name = strrep(tmp_name, 'tmp', 'bsp')
        save(strcat(tmp_path, tmp_name), 'tmp');
        save(strcat(bsp_path, bsp_name) , 'bsp');
    end


end
