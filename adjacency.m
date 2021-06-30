%%graph
h4 = WattsStrogatz(128,1,1);
plot(h4,'NodeColor','k','EdgeAlpha',1);
title('Watts-Strogatz Graph with $N = 500$ nodes, $K = 25$, and $\beta = 1$', ...
    'Interpreter','latex')
A = adjacency(h4);
FULL = full(A);
%%
colormap hsv
A = graph(FULL);
deg = degree(A);
nSizes = 3*sqrt(deg-min(deg)+0.2);
nColors = deg;
plot(A,'MarkerSize',nSizes,'NodeCData',nColors,'EdgeAlpha',0.5)
%title('Watts-Strogatz Graph with $N = 500$ nodes, $K = 25$, and $\beta = 0.15$',4...
%    'Interpreter','latex')
colorbar