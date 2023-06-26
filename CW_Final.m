clc; clear all; close all;

Data = readtable('Folds5x2_pp.xlsx');
% Rename AT to T and PE to EP
Data = renamevars(Data,["AT", "AP", "RH", "V", "PE"], ["T", "AP", "RH"...
    ,"V", "EP"]);

% removing all missing data.
Data = rmmissing(Data);

% Separating the columns into individual variables.
T = Data(:,"T");
V = Data(:,"V");
AP = Data(:,"AP");
RH = Data(:,"RH");
EP = Data(:,"EP");

T = table2array(T);
V = table2array(V);
AP = table2array(AP);
RH = table2array(RH);
EP = table2array(EP);

% Boxplots of the variables to search for anomalies  
subplot(3,2,1)
boxplot(T)
title("A box plot of T");
ylabel('Temperature degrees C')
subplot(3,2,2)
boxplot(V)
title("A box plot of V");
ylabel('Exhaust Vacuum cm Hg')

subplot(3,2,3)
boxplot(AP)
title("A box plot of AP");
ylabel('Ambient Pressure milibar')

subplot(3,2,4)
boxplot(RH)
title("A box plot of RH");
ylabel('Relative Humidity %')

subplot(3,2,5)
boxplot(EP)
title("A box plot of EP");
ylabel('Net hourly electrical energy output MW')

% Checking the ranges are right 
RangeT = range(T);
RangeV = range(V);
RangeAP = range(AP);
RangeRh = range(RH);
RangeEP = range(EP);
%All the ranges matach the website 

%%
% Calculating the mean
%
MeanT = mean(T);
MeanV = mean(V);
MeanAP = mean(AP);
MeanRH = mean(RH);
MeanEP = mean(EP);


%%
% Calculating the variance 
%
VarT = var(T);
VarV = var(V);
VarAP = var(AP);
VarRH = var(RH);
VarEP = var(EP);

%%
% Distribution of EP and features 
SDT = sqrt(VarT);
SDV = sqrt(VarV);
SDAP = sqrt(VarAP);
SDRH = sqrt(VarRH);
SDEP = sqrt(VarEP);


%%
% Hypothesis testing
%
%%
%Test on the mean 
%

%Can't use ztest due to the unknown mu and signma 

% size of the sample data used.
n = 956;
Sample_Data = datasample(EP,n);


Mean_sample = mean(Sample_Data);
root_n = sqrt(n);

mu = MeanEP;
Sample_variance = var(Sample_Data);
SD_sample = sqrt(Sample_variance);
v = n-1;

% h0 = mu = mean EP 
% h1: mu > mean EP
t0 = (Mean_sample - mu)/(SD_sample/root_n);
t_value = tinv(1-0.05,v);
% 1.3972 < 1.6465 which means accept H0 as t0 < tvalue  
% this means that that EP is equal to the true mean at the 5% sig level.
% We did one tail so that we can see if it's higher or lower than the mean.


%%
% Test on the variance 
%

% h0: signma = variance of EP
% h1: sigma < to variance of EP

Chi_squar = ((n-1)*Sample_variance)/VarEP; % equals 941.86
chi_value = chi2inv((1-0.05),v); % equals 1028
% as 941.89 < 1028 which indicates there is strong evidence to reject
% h0. The reason we have done one tailed is because we wanted to see if
% the value chi is 



%%
% Linear regression 
%


fullColumn = repmat('1',[9568,1]);
X = table(fullColumn);

Xtable = addvars(X,T,V,RH,AP);
X = table2array(Xtable);


% Separating the data into training and testing
% Creating the testing data 
rng('default');
cv_train = cvpartition(size(Xtable,1),'HoldOut',0.2);
idx = cv_train.test;
Train_DataX = Xtable(~idx,:);
Test_DataX  = Xtable(idx,:);
Train_DataX_array = table2array(Train_DataX);
Test_DataX_array = table2array(Test_DataX);

% Creating the training data 
rng('default');
cv_test = cvpartition(size(EP,1),'HoldOut',0.2);
idx = cv_test.test;
Train_DataEP = EP(~idx,:);
Test_DataEP  = EP(idx,:);


% Working out Beta 
Xtrain = Train_DataX_array';
%Xtrain = (Train_DataX_array.^2)';%uncomment when testing for x^2 
%Xtrain = (Train_DataX_array.^3)';%uncomment when testing for x^3
XtrainX = Xtrain*Train_DataX_array;
Beta = (inv(XtrainX))*(Xtrain*Train_DataEP);


% Working out the error from the testing data
Y_hat_test = Test_DataX_array*Beta;
e_test = Test_DataEP - Y_hat_test;
e_test_mean = mean(e_test);
% The model has a loss of 16.89% this is for normal x
% When x = x^2 the loss becomes 16.09%
% When x = x^3 the loss is 15.24%
% x = 0:1:1912;
% plot(x,Y_hat_test,'o')
% hold on 
% plot(x,Test_DataEP,'o')
% hold off

% working out the coeffiecient of determination R^2
SSr_test = sum((Y_hat_test - MeanEP).^2);
SSe_test = sum(e_test.^2);
Syy = SSr_test + SSe_test;
R_sqrt = SSr_test/Syy;
% the model accounts for 92.9% of the variability in the data. This is for 
% normal x.
% When x = x^2 the model accounts for 92.57%
% When x = x^3 the model accounts for 91.93%

% When increasing the power order of the model the accuracy of the model
% decreases and so does the loss. 


% T test for the intercept  
n_test = 1913;
k = 1;
Sigma_hat_sqr_Test = SSe_test/(n_test-k-1);

Mean_Test_DataX_1 = mean(Test_DataX_array(1));
Sxx_Test_1 = sum(Test_DataX_array(1).^2) - ((sum(Test_DataX_array(1)).^2)/n_test);

% H0: Beta(1) = 0
% H1: Beta(1) /= 0
T_test_intercept = (Beta(1))/sqrt(Sigma_hat_sqr_Test*((1/n_test)+((Mean_Test_DataX_1^2)/Sxx_Test_1)));
t_value_test_input = tinv(1-0.025,n_test-1);
% We reject H0 as 2.01 > 1.96. Suggesting there is no significant evidence 
% to suggest that the intercept is 0 at a 5% sig level 


% T test for T
Mean_Test_DataX_2 = mean(Test_DataX_array(2));
Sxx_Test_2 = sum(Test_DataX_array(2).^2) - ((sum(Test_DataX_array(2)).^2)/n_test);

% H0: Beta(T) = Beta
% H1: Beta =/ Beta
T_test_T = (Beta(2))/(sqrt(Sigma_hat_sqr_Test/Sxx_Test_2));%-21.2191
t_value_test_T = tinv(1-0.025,n_test-1); % 1.6457
% -21.2191 < 1.96 This means that we reject H0 as T_test_T < - t_value
% therefore, there is no significance evidence suggesting that Beta(T) is 
%equal to Beta at the 5% significance level.


% T test for V 
Mean_Test_DataX_3 = mean(Test_DataX_array(3));
Sxx_Test_3 = sum(Test_DataX_array(3).^2) - ((sum(Test_DataX_array(3)).^2)/n_test);

% H0: Beta = Beta 
% H1: Beta(V) < Beta
T_test_V = (Beta(3))/(sqrt(Sigma_hat_sqr_Test/Sxx_Test_3));% -2.5973 
t_value_test_V = tinv(1-0.05,n_test-1); % 1.6457
% -2.5973 < 1.6457 this means that we reject H0 as V_test_V < - t_value
% therefore, there is no significance evidence suggesting that Beta(V) is 
%equal to Beta at the 5% significance level.


% T test for RH 
Mean_Test_DataX_4 = mean(Test_DataX_array(4));
Sxx_Test_4 = sum(Test_DataX_array(4).^2) - ((sum(Test_DataX_array(4)).^2)/n_test);

% H0: Beta(RH) = Beta 
% H1: Beta(RH) < Beta
T_test_RH = (Beta(4))/(sqrt(Sigma_hat_sqr_Test/Sxx_Test_4));% -1.6570
t_value_test_RH = tinv(1-0.05,n_test-1); % 1.6457
% -1.6570 < 1.6457 this means that we reject H0 as RH_test_RH < - t_value
% therefore, there is no significiant evidence suggesting that Beta(RH) is 
%equal to Beta at the 5% significance level.


% T test for AP 
Mean_Test_DataX_5 = mean(Test_DataX_array(5));
Sxx_Test_5 = sum(Test_DataX_array(5).^2) - ((sum(Test_DataX_array(5)).^2)/n_test);

% H0: Beta(AP) = Beta 
% H1: Beta(AP) < Beta
T_test_AP = (Beta(5))/(sqrt(Sigma_hat_sqr_Test/Sxx_Test_5)); % 0.7522
t_value_test_AP = tinv(1-0.05,n_test-1); % 1.6457
% 0.7522 <1.6457 this means that we accept H0 as T_test_AP > -t_value 
% therefore, there is significant evidence to suggest that Beta(AP) is
% equal to Beta at the 5% significance level.



%%
% Logistic Regression 
%
%

% Setting the value of the training output to be between 0 and 1.
Train_EP = zeros(size(Train_DataEP));
for ii = 1:length(Train_DataEP)
    if Train_DataEP(ii)> MeanEP
        Train_EP(ii) = 1;
    else 
        Train_EP(ii) = 0;
    end 
end 

% Setting the value of the testing output to be between 0 and 1.
Test_EP = zeros(size(Test_DataEP));
for ii = 1:length(Test_DataEP)
    if Test_DataEP(ii)> MeanEP
        Test_EP(ii) = 1;
    else 
        Test_EP(ii) = 0;
    end 
end 


% alpha = learning rate 
alpha = 0.001;

% Variable for the number of Epochs I want the loop to run for. 
nn = 0:1:1000;

% Choosing the starting point for the gradient decent.
Beta_logistic = [0;0;0;0;0];

% Choose your input data here 
Input_data = Test_EP;

output = logistic(Test_DataX_array,MeanEP,alpha,Beta_logistic,Test_EP,...
    nn,Train_DataX_array,Train_EP,Input_data);

function [Beta_logistic,error_logistic_mean,R_sqrt_logistic,Y_hat_logstic,...
    output_logistic_input_data]= logistic(Test_DataX_array, ...
    MeanEP,alpha,Beta_logistic,Test_EP,nn,Train_DataX_array,...
    Train_EP,Input_data)
    

    % Calculating the gradient decent 
    for idx = 2:length(nn)
        
        %Train_DataX_array is a 7655x5 char 
        Beta_x = -(Beta_logistic(:,idx-1))'*Train_DataX_array(idx,:)';

        % Solving h_Beta 
        h_Beta = 1./(1+exp(Beta_x));
    
        % Solving for Beta(t+1)
        error_train = Train_EP' - h_Beta;       
        Adding_Beta_logistic= alpha*(sum(Train_DataX_array.*(error_train')));
   
        Beta_logistic(:,idx) = Beta_logistic(:,idx-1) + Adding_Beta_logistic';
    end 
    
    % Plotting the graph of Beta_logistic to find the optimum value of
    % Beta.
    figure    
    plot(nn,Beta_logistic)
    title('Value of Beta against the number of Epoch')
    xlabel('Epoch')
    ylabel('Beta')
    hold off

    % Working out the output using the optimised Beta value and then setting
    % it to be between 0 and 1 depending on the value.
    Y_hat_logistic = Test_DataX_array*Beta_logistic(:,1000);
            h_Beta_normalised_logistic = zeros(size(Y_hat_logistic));
        for ii = 1:length(Y_hat_logistic)
            if Y_hat_logistic(ii)> 0.5
                output_logistic(ii) = 1;
            else 
                output_logistic(ii) = 0;
            end 
        end 


    % Working out the error rate     
    error_logistic = Test_EP - output_logistic';
    error_logistic_mean = mean(error_logistic);
    %27% at 0.05 learning rate 

    % Working out the coeffiecient of determination R^2
    SSr_test_logistic = sum((output_logistic - MeanEP).^2);
    SSe_test_logistic = sum(error_logistic.^2);
    Syy_logistic = SSr_test_logistic + SSe_test_logistic;
    R_sqrt_logistic = SSr_test_logistic'/Syy_logistic';
    % 100% at 0.05 learning rate  

    % Working out the output using the optimised Beta value and then setting
    % it to be between 0 and 1 depending on the value.
    Y_hat_logistic_input = Input_data*Beta_logistic(:,1000)';
            h_Beta_normalised_logistic_input = zeros(size(Y_hat_logistic));
        for ii = 1:length(Y_hat_logistic_input)
            if Y_hat_logistic_input(ii)> 0.5
                output_logistic_input_data(ii) = 1;
            else 
                output_logistic_input_data(ii) = 0;
            end 
        end 
    
        
end 


