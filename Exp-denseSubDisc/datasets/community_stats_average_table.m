addpath('~/GitHubRepos/hkgrow/')
addpath('../pprpush')
addpath('~/data/snap-top10')

%%
clc
datasets = {'DBLP','Amazon','LiveJournal','Orkut','Youtube'};
percentofset = 5;
for graph = datasets
   
  load(strcat(char(graph),'-top10.mat'))
  outputfile = strcat('Output',num2str(percentofset),'/',char(graph),'_hkpr.txt');
  outputmatrix = strcat('Output',num2str(percentofset),'/',char(graph),'_hkpr.mat');
  
  load(strcat(char(graph),'-seed-starter.mat'))
  
  if percentofset == 2
      S = S2;
  elseif percentofset == 3
      S = S3;
  else
      S = S5;
  end
  
  comm = C;
  n = full(size(C,1));
  m = full(nnz(A)/2);
  d = sum(A,1)';
  volA = sum(nonzeros(A));
  n = size(A,1);
  numcom = size(comm,2);
  fid = fopen(outputfile,'w');
  
  sizes = round(mean(comsizes));
  conds = mean(Conds);
  
  for commID = 1:numcom
      
     Target = find(comm(:,commID));
     Rstart = find(S(:,commID));
     R = neighborhood(A,Rstart,1);
     [cutR,volR,edgesR,condR] = set_stats(A,R,volA);
     [pR,rR,fR] = AdjustedPRF(Target,R,[]);
    precisions(commID) = pR;
    recalls(commID) = rR;
    f1s(commID) = fR;
  end
  pr = mean(precisions);
  re = mean(recalls);
  f1s = mean(f1s);
     
  %fprintf(' %s & %d & %d & %d & %.4f \\\\\n',char(graph),full(n),full(m),full(sizes),full(conds))
    fprintf(' %s & %d & %d & %d & %.4f %.4f & %.4f & %.4f \\\\\n',char(graph),full(n),full(m),full(sizes),full(conds),pr,re,f1s)
end

