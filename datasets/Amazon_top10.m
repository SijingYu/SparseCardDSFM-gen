%addpath('~/GitHubRepos/LargeFiles/')

% Amazon is a little different in that it has a couple repeated communities
% We will remove the repeated communities here

tic
load com-Amazon.mat
toc
A = Problem.A;
C = Problem.aux.Communities_top5000;
comsizes = sum(C);

% Arrange them by size
[comsizes,order] = sort(comsizes);
C = C(:,order);
num = 50;
Cbig = C(:,end-num+1:end);

% Hard-code in the top-10 sized distinct communities
Top10distinct = [37, 38,41:45,47:49];
comfound = 0;
C = Cbig(:,Top10distinct);
comsizes = full(sum(C,1)');
%for i = 1:10
    
%   comm = find(C(:,i));
   
%   [~,~,~,cond] = set_stats(A,comm);
%   Conds(i) = cond;
%   fprintf('AMAZON: Community %d has %d nodes and conductance %f \n',i,numel(comm),cond)
    
%end

save('Amazon-top10.mat','C','A','comsizes')