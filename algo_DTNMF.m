%% sum_v=1^V av^2 ||PvHv - GvFv'||_F^2  + lambda ||Pv||_F^2 + beta ||F^||_Sp^p 
%% s.t £¨Hv)_tv = (X_v)_tv ;  Pv(HvHv¡¯+ lambda* I)Pv'= I, Gv'Gv = Ik,   Fv >=0. a1= 1;
%% Incomplete Multiview Clustering using Discriminative Feature Recovery and Tensorized Matrix Factorization

function [Index,obj,res_max,T_alpha] = algo_DTNMF(X, label ,m, lambda, beta,p ,ind)
% labels : ground truth   n *1.
% lambda : the hyper-parameter.
% beta   : the hyper-parameter.
% p      : the parameter of tensor learnint
% ind    £º missing index : n *view
% m      : the dimension of projection

% Xv     : incomplete data with zero pedding, dv *n  
% Gv     : m *k
% Fv     : k *n 
% Pv     : m * dv 

nV = length(X); N = size(X{1},2);
k = length(unique(label));
weight_vector = ones(1,nV)';        % the defult weight_vector of tensor Schatten p-norm

%% ==============Variable Initialization=========%%
for iv = 1:nV
    dim = size(X{iv},1);
    P{iv} = zeros(m, dim);    % projection
    Y{iv} = zeros(N, k);       % lagrange
    J{iv} = zeros(N, k);       % auxiliary variable
    H{iv} = X{iv}; 
    G{iv} = zeros(m, k);
    F{iv} = zeros(N, k);
    Q{iv} = zeros(N, k);   % solve F
    QQ{iv}= zeros(N, k);   % solve J
    PQ{iv}= zeros(N,k);    % solve P
end
alpha = ones(1,nV)/nV;
T_alpha = zeros(nV,200);
T_alpha(:,1) = alpha;
%%
% disp('--------------Anchor Selection and Bipartite graph Construction----------');
% tic;
% opt1. style = 1;
% opt1. IterMax =50;
% opt1. toy = 0;
% [~, B] = My_Bipartite_Con(X,cls_num,0.5, opt1,10);
% toc;

%% =====================  Initialization =====================
sX = [N, k, nV];
Isconverg = 0; iter = 1;
rho = 1e-4; max_rho = 10e12; pho_rho = 1.1;   % penalty factor.;  upper bound,; update step
Pstops = 10e-3;

%% =====================Optimization=====================
while(Isconverg == 0)
    %% solve G{v}
     for iv =1:nV
         part1 = P{iv} * H{iv} * F{iv};
        [Unew,~,Vnew] = svd(part1,'econ');
        G{iv} = Unew*Vnew';
    end
    %% solve F{v}
    for iv =1:nV
        Q{iv} = (J{iv} - Y{iv}/rho);  %% 
        temp = (alpha(iv)^2 * H{iv}'*P{iv}'*G{iv} + rho/2 * Q{iv})./(alpha(iv)^2 + rho/2);
        F{iv} = max(temp,0);
    end
    

    %%  solve J{v}
    for iv =1:nV
        QQ{iv}=(F{iv} + Y{iv}/rho);
    end
    Q_tensor = cat(3,QQ{:,:});
    Qg = Q_tensor(:);
    [myj, ~] = wshrinkObj_weight_lp(Qg, beta* weight_vector./rho,sX, 0,3,p);
    J_tensor = reshape(myj, sX);
    for k=1:nV
        J{k} = J_tensor(:,:,k);
    end
  
    %% solve P{v}
    for iv =1: nV
        linshi_St = H{iv}* H{iv}'+  lambda/alpha(iv)^2 * eye(size(H{iv},1)); 
        St{iv} = mpower(linshi_St,-0.5);
        St2{iv} =  St{iv}*H{iv}*F{iv}*G{iv}';
        [U,~,V] = svd(St2{iv},'econ');
        P{iv} =V*U'*St{iv};
    end

    %% solve H{v}
    for iv = 1:nV
%         St3{iv} = P{iv}'* P{iv}+ eps* eye(size(H{iv},1));
         St3{iv} = P{iv}'* P{iv};
         St4{iv} = pinv(St3{iv});
        tempH = St4{iv} * P{iv}'* G{iv} * F{iv}';
        index = find(ind(:,iv)');
        tempH(:,index) = X{iv}(:,index);
        H{iv} = tempH;
    end    
      
    %% solve av
    M = zeros(nV,1);
    for iv = 1:nV
        M(iv) = norm( P{iv}*H{iv} - G{iv} * F{iv}','fro')^2 ;
    end
    Mfra = M.^-1;
    Qa = 1/sum(Mfra);
    alpha = Qa*Mfra;
    T_alpha(:,iter+1) = alpha;  
    
    %% solve Y and  penalty parameters
    for iv=1:nV
        Y{iv} = Y{iv} + rho*(F{iv}-J{iv});
    end
    rho = min(rho*pho_rho, max_rho);

    %% compute loss value
    term1 = 0;
    for iv = 1:nV
        term1 = term1 + alpha(iv)^2 * M(iv);
        res(iv) = norm(F{iv}-J{iv}, inf );
    end
    obj(iter) = term1;
    res_max(iter) = max(res);

    %% ==============Max Training Epoc==============%%
    if (iter>10 && ( res_max(iter)< Pstops))                  %% original
%   if (iter>10 && ( res_max(iter)< Pstops))  || iter  > 50   % only for fmnist
        Isconverg  = 1;
        SumF = 0;
        Sa = 0;
        for iv = 1: nV
            SumF = SumF + alpha(iv)^2 *F{iv};
            Sa = Sa + alpha(iv)^2;
        end
        SumF = SumF/Sa;
       [~,Index] = max(SumF,[],2);
       res = Clustering8Measure(label, Index);     
    end
    iter = iter + 1;
end
