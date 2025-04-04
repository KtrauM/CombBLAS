% for titan
% class = 'ER' or  'G500' or 'SSCA'
% scale = number of rows/cols = 2^scale
function batchGen(nodes, class, scale, deg)

maxCore = nodes* 16;
fileName = sprintf('batch_%s_%d_%d', class, scale, maxCore);
fileID = fopen(fileName,'w');
fprintf(fileID,'#!/bin/bash\n');
fprintf(fileID,'#PBS -A CSC103\n');
fprintf(fileID,'#PBS -q debug\n');
fprintf(fileID,'#PBS -l nodes=%d\n', nodes);
fprintf(fileID,'#PBS -l walltime=00:59:00\n');
fprintf(fileID,'#PBS -N spGEMMexp_%s_%d_%d\n', class, scale, maxCore);
fprintf(fileID,'#PBS -j oe\n');
fprintf(fileID,'cd $MEMBERWORK/csc103\n');



% problem specific stats
%scale = 24;
%class = 'ER';
% deg = 16;


% machine specifc infp 
% for titan

layers = [1,2,4,8,12,16];
threads = [1,2,4,8,16];
coresPerNode = 16;
coresPerSocket = 8;

for t = threads
    fprintf(fileID, '\nexport OMP_NUM_THREADS=%d\n', t);
    if(t==coresPerSocket) 
        cc = 'numa_node';
    else
        cc = 'depth';
    end
        
    for c = layers
        dim1 = floor(sqrt(maxCore/(t*c)));
        dim2 = dim1;
        ncores = dim1*dim2*c*t;
        nprocs = dim1*dim2*c;
        N = coresPerNode/t;
        S = coresPerSocket/t;
        if(t<=coresPerSocket)
            fprintf(fileID,'aprun -n %d -d %d -N %d -S %d -cc %s ./mpipspgemm %d %d %d %s %d %d column\n', nprocs, t, N, S, cc, dim1, dim2, c, class, scale, deg);
        else
            fprintf(fileID,'aprun -n %d -d %d -N %d ./mpipspgemm %d %d %d %s %d %d column\n', nprocs, t, N, dim1, dim2, c, class, scale, deg);
        %fprintf(fileID,'%d\t %d\t %d\t %d\t %d\t %d\t\n', ncores, nprocs, dim1, dim2, c, t);
        end
    end
end

fclose(fileID);
    