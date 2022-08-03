function H = generate_H_3d(Nx, Ny, w, step, z_val)

%Nx is the the number of columns in the heart grid
%Ny is the the number of rows in the heart grid
%w indicates how wider than the heart grid the torso grid is eg. if w=10
%in the heart grid of (0,100)(0,100) torso grid is (-10,110)(-10,110)
%step is jumps over the heart nodes in the grid

% nn is the number of nodes in the torso grid
nn = length(-w:10:Nx+w)*length(-w:10:Ny+w);

%initializing H
H = zeros(nn, Nx*Ny);

rows = [1, 10:10:100];
cols = [1, 10:10:100];
count = 0;

for r = 1:11
    for c = 1:11
        j = rows(r);
        i = cols(c);
        count = count + 1;
%         grid_point is the current node in the torso grid (vector form is
%         used to calculate the distance with all the heart nodes in the
%         same loop iteration for a faster computation)
        grid_point = [ones(Nx*Nx,1)*j, ones(Ny*Ny,1)*i, ones(Ny*Ny,1)*z_val];
        
%         heart indices
        heart = (0:1:(Nx*Ny)-1)';
%         considering that heartcoordinates start from (0,0) get the x and
%         y of the heart nodes from the indices
        heart_Y = fix(heart/Ny);
        heart_X = rem(heart,Nx);
        heart_z = zeros(Nx*Ny, 1);
        heart_point = [heart_X, heart_Y, heart_z];
        
%         compute the heart torso node distance
        D = 1./sqrt((heart_point(:,1) - grid_point(:,1)).^2 + (heart_point(:,2) - grid_point(:,2)).^2 + (heart_point(:,3) - grid_point(:,3)).^2);
        
%         if D is infinity it means that the torso node has the same coordinates with the heart node so set D to 1
%         D(D == inf) = 1;
        H(count,:) = D;
        H(count,:) = H(count,:)/sum(D);

    end
end


end