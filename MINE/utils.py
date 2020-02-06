def calculate_EMA(prev_value, curr_value, weight):
    assert weight >= 0 and weight <= 1
    return weight*curr_value + (1-weight)*prev_value
