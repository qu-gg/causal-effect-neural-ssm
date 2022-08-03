function mea_n = gen_noise(mea, SNR)
% apply the random generation function in matlab toolbox
% noise = dim1*dim2matter
% mea = load('final-data/simulation_results/output/ymat.mat')
% mea = mea.mea ;
% mea = H*U;
type = 'Gau';
PARA = 1; 
flag = 0;
[dimz,t] = size(mea);
if(SNR==0);
    fid = fopen('Measurement_noisy.bin','wb');
    fwrite(fid,mea,'double');
    fclose(fid);
    fid = fopen('Noise_mea.bin','wb');
    fwrite(fid,zeros(t, 1), 'double');
    fclose(fid);
elseif flag==0;   % noise differes at each node, but should time invariant statistics
    sigPow = sum(mea .* mea,2) / t;
    noiPow_mean = sigPow ./ 10^(SNR / 10);
    noiPow_SD = sqrt(noiPow_mean);
    for i=1:dimz;
        noise(i,:) = normrnd(0, noiPow_SD(i), 1, t);
    end
    %noiPow_SD = noiPow_mean;
else    % noise differes at each time instant, but should statistically invariant for each node at each time instant
    sigPow = sum(sum(mea .* mea)) / (dimz * t);
    noiPow_mean = sigPow ./ 10^(SNR / 10);
    noiPow_SD = noiPow_mean + noiPow_mean / 100 * randn(1, t);
    if type == 'Gau';
		mean = 0; var = 1;
		noise = normrnd(mean, var, dimz, t);
    elseif type == 'Poi'
		[mean,var] = poisstat(PARA);
		noise = poissrnd(PARA, dimz, t);
    elseif type =='Exp'
		[mean,var] = expstat(PARA);
		noise = exprnd(PARA, dimz, t);
    elseif type == 'Ray'
		[mean,var] = raylstat(PARA);
		noise = raylrnd(PARA, dimz, t);
    elseif  type =='Uni'
		[mean,var] = unifstat(0,1);
		noise = unifrnd(0,1,dimz,t);
    elseif type == 'Geo'
		[mean,var] = geostat(PARA);
		noise = geornd(PARA, dimz, t);
    end
  	%normalize-- to mean=0,var=1
	noise = (noise - mean) / sqrt(var);
    % changing to var = noise_power
    noise = (ones(dimz, 1) * sqrt(noiPow_SD)) .* noise;
end 
    mea_n = mea + noise;

end
