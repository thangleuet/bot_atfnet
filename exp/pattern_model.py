class PatternModel():
    def __init__(self, list_zigzag_pred, list_zigzag_true, index_candle, gt, pred, actual, path_image, x_zigzag_data_true, y_zigzag_data_true, x_zigzag_data_pred, y_zigzag_data_pred):
        self.list_zigzag_pred = list_zigzag_pred
        self.list_zigzag_true = list_zigzag_true
        self.confirm_count = 0
        self.index_candle = index_candle
        self.gt = gt
        self.pred = pred
        self.actual = actual
        self.path_image = path_image
        self.x_zigzag_data_true = x_zigzag_data_true
        self.y_zigzag_data_true = y_zigzag_data_true
        self.x_zigzag_data_pred = x_zigzag_data_pred
        self.y_zigzag_data_pred = y_zigzag_data_pred