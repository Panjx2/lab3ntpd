from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None

def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) == len(y_test)

def test_predictions_value_range():
    preds, _ = train_and_predict()
    assert all(p in [0, 1, 2] for p in preds)

def test_model_accuracy():
    preds, y_test = train_and_predict()
    acc = get_accuracy(y_test, preds)
    assert acc >= 0.7
