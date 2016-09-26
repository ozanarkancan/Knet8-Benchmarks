require 'torch'

local data_loader = {}

function data_loader.load_data(th_file, train_perc)

    local data = {}
    local raw_data = torch.load(th_file)

    -- Shuffle Data
    local indexes = torch.randperm(raw_data:size(1)):long()
    raw_data = raw_data:index(1,indexes)

    -- Index number to split data into train & test.
    if train_perc == 100 then
       train_test_split_index = raw_data:size(1)
    else
       train_test_split_index = (raw_data:size(1) / 100) * train_perc
    end
    -- The following are the column names of the data.
    -- Variables in order:
    -- #        row number
    -- CRIM     per capita crime rate by town
    -- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    -- INDUS    proportion of non-retail business acres per town
    -- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    -- NOX      nitric oxides concentration (parts per 10 million)
    -- RM       average number of rooms per dwelling
    -- AGE      proportion of owner-occupied units built prior to 1940
    -- DIS      weighted distances to five Boston employment centres
    -- RAD      index of accessibility to radial highways
    -- TAX      full-value property-tax rate per $10,000
    -- PTRATIO  pupil-teacher ratio by town
    -- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    -- LSTAT    % lower status of the population
    -- MEDV     Median value of owner-occupied homes in $1000's (target column)

    -- target column is the last one.
    t_coulmn = raw_data:size(2)

    -- cloning targets because the data will be standardised after.
    data.train_targets = raw_data[{ {1,train_test_split_index},{t_coulmn} }]:clone()
    if train_perc ~= 100 then
       data.test_targets =  raw_data[{ {train_test_split_index + 1,raw_data:size(1)},{t_coulmn} }]:clone()
    end
    -- Standardize Data.
    local std = std_ or raw_data:std()
     local mean = mean_ or raw_data:mean()
    raw_data:add(-mean)
     raw_data:mul(1/std)

    local dimensions = raw_data:size(2)-1

    -- input data -> we ignore the first column as it's just the row number.
    data.train_data = raw_data[{ {1,train_test_split_index},{2,dimensions} }]
    if train_perc ~= 100 then
       data.test_data = raw_data[{ {train_test_split_index + 1,raw_data:size(1)},{2,dimensions} }]
    end
    return data
end


return data_loader