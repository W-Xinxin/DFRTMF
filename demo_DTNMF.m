%% sum_v=1^m av^2 ||PvHv - GvFv||_F^2  + lambda ||Pv||_F^2 + beta ||F^||_Sp^p 
%% s.t £¨Hv)_tv = (X_v)_tv ;  Pv(HvHv¡¯+ lambda* I)Pv'= I, Gv'Gv = Ik,   Fv >=0. a1= 1;
%% Discriminative and Tensorized non-negative matrix factorization for incomplete multiview clustering(DTNMF)

clear;
clc;

addpath(genpath('./'));

resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/'));
end

datadir='/home/viplab/Desktop/datasets/MyIncompleteData/';

dataname={'MSRCv1'};

numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};

for idata = 1 
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
    for dataIndex = 1
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        cls_num = length(unique(gt));
        k= cls_num;
        tic;
        [X1, ind] = findindex(data, index);

        time1 = toc;
        maxAcc = 0;
        TempLambda= [1e-5,1e-4,1e-3,1e-2,1e-1];
        TempLambda= [1e-2];
        TempBeta =[1];   
        TempP= [0.1:0.1:1];
        TempP = [0.1];  % for low rank tensor 
        m = k;             % for projection: k+1, only wiki is k
        
        ACC = zeros(length(TempLambda),length(TempBeta),length(TempP));
        NMI = zeros(length(TempLambda), length(TempBeta),length(TempP));
        Purity = zeros(length(TempLambda), length(TempBeta),length(TempP));
        idx = 1;
            for LambdaIndex1 = 1 : length(TempLambda)
             lambda1 = TempLambda(LambdaIndex1);  
             for LambdaIndex2 = 1 : length(TempBeta) 
              lambda2 = TempBeta(LambdaIndex2);  
              for LambdaIndex3 = 1 : length(TempP)
                p = TempP(LambdaIndex3);
                disp([char(dataname(idata)), char(numname(dataIndex)),'-l1=', num2str(lambda1),'-b2=', num2str(lambda2) '-p=', num2str(p)]);
                tic;
                para.c = cls_num; % K: number of clusters
                rand('seed',6666);
                [PreY,obj,res_max, Talpha] = algo_DTNMF(X1,gt,m,lambda1, lambda2,p,ind); % X,Y,lambda,d,numanchor
                time2 = toc;
                
                tic;
                for rep = 1 : 10
                    res(rep, : ) = Clustering8Measure(gt, PreY);
                end
                time3 = toc;

                runtime(idx) = time1 + time2 + time3/10; 
                disp(['runtime:', num2str(runtime(idx))])
                idx = idx + 1;
                tempResBest(dataIndex, : ) = mean(res);
                tempResStd(dataIndex, : ) = std(res);
                ACC(LambdaIndex1, LambdaIndex2,LambdaIndex3) = tempResBest(dataIndex, 1);
                NMI(LambdaIndex1, LambdaIndex2,LambdaIndex3) = tempResBest(dataIndex, 2);
                Purity(LambdaIndex1, LambdaIndex2,LambdaIndex3) = tempResBest(dataIndex, 3);
                save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1),'-b2=', num2str(lambda2),'-p=', num2str(p), ...
                    '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd','obj','res_max','Talpha');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end
               end
              end
            end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC(:))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'PResBest', 'PResStd');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end
