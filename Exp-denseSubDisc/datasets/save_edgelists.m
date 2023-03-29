FileList = {'orkut','livejournal','wiki-undir'};
FileList = {'Amazon'};
for z = 1:numel(FileList)
    

    file = strcat(char(FileList(z)),'.edgelist');
    fid = fopen(file, 'w');

    load(strcat(char(FileList(z)),'-communities.mat'))

    [a,b,v] = find((A));

    for i = 1:numel(a)
        fprintf(fid,'%d %d\n',a(i) - 1, b(i) -1);
    end

    fclose(fid)
    
end
