datasets = {'Amazon'}
for graph = datasets
   
  load(strcat(char(graph),'-top10.mat'))  

  [n,numcom] = size(C);
  S2 = zeros(n,numcom);
  S3 = zeros(n,numcom);
  S5 = zeros(n,numcom);

  for commID = 1:numcom
      
    Target = find(C(:,commID));
    TarSize = numel(Target);
     
    % Get a different permutation every time so the sets don't look too
    % similar
    n2 = round(TarSize*.02);
    p = randperm(TarSize);
    R2 = Target(p(1:n2));
     
    n3 = round(TarSize*.03);
    p = randperm(TarSize);
    R3 = Target(p(1:n3));
          
    n5 = round(TarSize*.05);
    p = randperm(TarSize);
    R5 = Target(p(1:n5));
    
    S2(R2,commID) = 1;
    S3(R3,commID) = 1;
    S5(R5,commID) = 1;
  end
  
  save(strcat(char(graph),'-seed-starter.mat'),'S2','S3','S5')
end
     
     