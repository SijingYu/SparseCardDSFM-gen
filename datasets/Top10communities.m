addpath('~/GitHubRepos/LargeFiles/')

graph = {'DBLP','LiveJournal','Orkut'};

for g = graph
 
    tic
    load(strcat('com-',char(g),'.mat'))
    loading = toc;
    fprintf('Took %f seconds to load %s \n',loading,char(g));

    A = Problem.A;
    C = Problem.aux.Communities_top5000;
    comsizes = sum(C);

    % Arrange them by size
    [comsizes,order] = sort(comsizes);
    C = C(:,order);
    num = 10;
    Cbig = C(:,end-num+1:end);

    Conds = zeros(num,1);
    for i = 1:num

        % Get the original community
        comm = find(Cbig(:,i));
   
        % Take its largest component
        Ac = A(comm,comm);
        [ci,comps] = graphconncomp(Ac);
        comnew = comm(comps==mode(comps));
        
        if ci > 1
            fprintf('\tCommunity %d had multiple components \n',i)
        end
       [~,~,~,cond] = set_stats(A,comnew);
       fprintf('%s: Community %d has %d nodes and conductance %f \n',char(g),i,numel(comnew),cond)
       Conds(i) = cond;
       
    end
    C = Cbig;
    comsizes = sum(C,1)';
    size(comsizes)

save(strcat(char(g),'-top10.mat'),'C','A','Conds','comsizes')
end

