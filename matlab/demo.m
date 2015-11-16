%% Load the data

load('multipie4030')

%%

initial = find(gnd(:, 1) < 10 & gnd(:, 4) == 3);
newset  = find(gnd(:, 1) >= 10 & gnd(:, 4) == 3);
% gnd(:, 1): Identity
% gnd(:, 2): Pose
% gnd(:, 3): Emotion
% gnd(:, 4): Illumination

misc.Xr = newset;
misc.Y_test = {gnd(newset, 2), gnd(newset, 3), gnd(newset, 1)};     % The testing labels
misc.Y_train = {gnd(initial, 2), gnd(initial, 3), gnd(initial, 1)}; % The training labels


X_train = matrix(:, initial);
[Ztransfer, Htransfer] = ssdeep_seminmf(X_train, [100 100 100], {gnd(initial, 2), gnd(initial, 3), gnd(initial, 1)}, misc, 'maxiter', 400, 'lambda', [0.01 0.01 0.001], 'cache', 0);

%% With transfer weights

[Znew, Hnew] = deep_seminmf(matrix(:, newset), [100 100 100], 'maxiter', 100, 'cache', 0, 'z0', Ztransfer);

%% Without transfer weights

[Ztrain, Htrain] = deep_seminmf(X_train, [100 100 100], 'maxiter', 500, 'cache', 0);


%% Evaluate Clustering
h = Hno;
names = {'Identity', 'Pose', 'Emotion'};

fprintf('\n');
for i = 1:3;
    fprintf('%s: ', names{i});
....
    for j = 1:3;
        ac = evalResults(h{j}, gnd(newset, i));
        fprintf(1, '%.2f | ', ac);
    end
    fprintf(1, '\n');
end
%% Classification

test_set = [];

for i = 1:max(gnd(:, 1));
    ind = find(gnd(:, 1) == i & gnd(:, 4) == 3);
    ind = ind(randperm(numel(ind), 40));
    
    test_set = [test_set ; ind];
end


train_set = setdiff(find(gnd(:, 4) == 3), test_set);
matrix = max(matrix/max(matrix(:)), eps);

%%

% gnd(:, 1): Identity
% gnd(:, 2): Pose
% gnd(:, 3): Emotion
% gnd(:, 4): Illumination

misc.Xr = matrix(:, test_set);
misc.Y_test = {gnd(test_set, 2), gnd(test_set, 3), gnd(test_set, 1)};     % The testing labels
misc.Y_train = {gnd(train_set, 2), gnd(train_set, 3), gnd(train_set, 1)}; % The training labels



X_train = matrix(:, train_set);
[Ztrain, Htrain] = ssdeep_seminmf(X_train, [100 100 100], misc.Y_train, misc, 'maxiter', 1000, 'lambda', [0.01 0.01 .001], 'cache', 0);


%% Evaluate Classification
h = Hno;
names = {'Identity', 'Pose', 'Emotion'};

fprintf('\n');
for i = 1:3;
    fprintf('%s: ', names{i});
....
    for j = 1:3;
%          mdl = train(misc.Y_train{i}, sparse(reshape(cell2mat(Htrain), 1200, [])'), '-q');
        mdl = train(misc.Y_train{i}, sparse(Htrain{j}'), '-q');
        D = Ztrain{1};
        for k = 2:j
            D = D * Ztrain{k};
        end
        
        
        Hr = pinv(D) * matrix(:, test_set);
        [~, ac, ~] = predict(misc.Y_test{i}, sparse(Hr'), mdl, '-q');
        
        fprintf(1, '%.2f | ', ac(1));
    end
    fprintf(1, '\n');
end
%% wsnmf multilabel

misc.Xr = matrix(:, test_set);
misc.Y_test = {gnd(test_set, 2), gnd(test_set, 3), gnd(test_set, 1)};     % The testing labels
misc.Y_train = {gnd(train_set, 2), gnd(train_set, 3), gnd(train_set, 1)}; % The training labels



X_train = matrix(:, train_set);
[Ztrain, Htrain] = wsnmf_ma(X_train, misc.Y_train, 100, 'maxiter', 500, 'lambda', {0.01 0.01 0.01});

%% Evaluate Classification for wsnmf
h = Htrain; 
names = {'Pose', 'Emotion', 'Identity'};

fprintf('\n');
for i = 1:3;
    fprintf('%s: ', names{i});
    mdl = train(misc.Y_train{i}, sparse(Htrain'), '-q');

    D = Ztrain; 

    Hr = pinv(D) * matrix(:, test_set);
    [~, ac, ~] = predict(misc.Y_test{i}, sparse(Hr'), mdl, '-q');

    fprintf(1, '%.2f | ', ac(1));
    
    fprintf(1, '\n');
end