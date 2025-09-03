from sklearn.metrics import mean_absolute_error

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
