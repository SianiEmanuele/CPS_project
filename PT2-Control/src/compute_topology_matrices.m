function [Ad, L, G] = compute_topology_matrices (topology_type) 

switch topology_type
    case 1
        % Topology 1:
        % Leader talks to S1, S1 talks to all the others directly
        Ad = [
            0 0 0 0 0 0;
            1 0 0 0 0 0;
            1 0 0 0 0 0;
            1 0 0 0 0 0;
            1 0 0 0 0 0;
            1 0 0 0 0 0;
        ]; 

        G = diag([1 0 0 0 0 0]);

    case 2
        % Topology 2:
        % Leader talks (through S1) to a complete graph of followers
        Ad = [
            0 1 1 1 1 1;
            1 0 1 1 1 1;
            1 1 0 1 1 1;
            1 1 1 0 1 1;
            1 1 1 1 0 1;
            1 1 1 1 1 0;
        ]; 
        G = diag([1 0 0 0 0 0]);

    case 3
        % Topology 4 - baseline:
        % Leader talks to all the followers

        Ad = [
            0 1 0 0 0 0;
            0 0 1 0 0 0;
            0 0 0 1 0 0;
            0 0 0 0 1 0;
            0 0 0 0 0 1;
            1 0 0 0 0 0;
        ]; 

        G=diag([1 1 1 1 1 1]);
    otherwise
        fprintf("There is no topology identified as number %d!\n", topology_num);
end


in_degrees = sum(Ad,2);
L = diag(in_degrees);


%%%%%%%%%%%%%%%%%%%%%%%%%% PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = size(Ad, 1);
Ad_aug = zeros(N+1);

% N.B: For plotting matlab takes the transpose of the Adj matrix we use in
% control theory
Ad_aug(2:end, 2:end) = Ad';
Ad_aug(1, 2:end) = diag(G)'; 
g_aug = digraph(Ad_aug);

% 3. Plotting
figure;
h = plot(g_aug);

%%%%%%%%%%%%% Layout fixes for prettier plot %%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. Relabel Nodes
h.NodeLabel = arrayfun(@(x) num2str(x), 0:N, 'UniformOutput', false);

if topology_type == 3 | topology_type == 2
    % Manual Layout: Leader in Middle
    x_coords = zeros(1, N+1);
    y_coords = zeros(1, N+1);
    
    % Node 1 (Leader) at center (0,0)
    x_coords(1) = 0; 
    if topology_type == 2
        y_coords(1) = 3;
    else
    y_coords(1) = 0;
    end
    
    % Followers arranged in a circle
    theta = linspace(0, 2*pi, N+1); 
    theta(end) = []; % Remove duplicate end point
    radius = 2; 
    angle_offset = pi/2; % Rotates circle so the first node is at the top
    
    for i = 1:N
        % i+1 maps to the specific follower node index
        x_coords(i+1) = radius * cos(theta(i) + angle_offset);
        y_coords(i+1) = radius * sin(theta(i) + angle_offset);
    end
    
    h.XData = x_coords;
    h.YData = y_coords;
    axis equal; % Keeps the circle round
    axis off; 
else
    % Standard Layout for other topologies
    layout(h, 'layered', 'Sources', 1);
end
% --------------------------

% Highlight Node 0 (Leader, internal index 1)
highlight(h, 1, 'NodeColor', 'r', 'MarkerSize', 8);

% Highlight edges coming from Node 0
leader_edges = outedges(g_aug, 1);
highlight(h, 1, 'Edges', leader_edges, 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');

title(['Augmented Topology ' num2str(topology_type) ' (Node 0 is Leader)']);
end