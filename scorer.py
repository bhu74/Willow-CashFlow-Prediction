import csv

def load_values(filename):
    ret = {}
    cast_failed = 0
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'Version' in row and row['Version'] != 'ACT':
                continue

            element = row['Account-Mapping']
            market = row['Market-Mapping']
            date = row['Year-mapping'] + '-' + row['Month-mapping']
            try:
                value = float(row['Adj_Value_Sum'])
            except:
                cast_failed += 1
                value = 0
            ret[(element, market, date)] = value
            data.append((element, market, date, value))

    # roll ups
    for (element, market, date, value) in data:
        # region level
        region_key = (element, market[0], date)
        ret[region_key] = ret.get(region_key, 0) + value
        # element level
        element_key = (element, date)
        ret[element_key] = ret.get(element_key, 0) + value
        # total level
        total_key = (date)
        ret[total_key] = ret.get(total_key, 0) + value
        
    if cast_failed > 0:
        print('Warning!', filename, cast_failed)
    return ret


def evaluate(truth_filename, pred_filename):
    truth = load_values(truth_filename)
    pred = load_values(pred_filename)

    if len(pred) != len(truth):
        print('Wrong # of Predictions!', pred_filename)
        return -1

    raw_scores = []

    for key, value in pred.items():
        if key not in truth:
            return -2
        MAPE = abs(truth[key] - value) / abs(truth[key])
        raw_score = max(0, 1 - MAPE)
        raw_scores.append(raw_score)

    avg = sum(raw_scores) / len(raw_scores)
    return avg * 100
