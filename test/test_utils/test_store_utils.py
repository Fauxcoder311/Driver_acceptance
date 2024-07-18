import json
import os
import pickle
import unittest

import pandas as pd
from nose.tools import raises
from sklearn.linear_model import LogisticRegression

from src.utils.store import InvalidExtension, Store


class TestStoreUtils(unittest.TestCase):
    def tearDown(self):
        for f in ("test.csv", "test.json", "test.pkl"):
            if os.path.isfile(f):
                os.remove(f)

    @raises(InvalidExtension, FileNotFoundError)
    def test_store_get_failures(self):
        self.assertRaises(InvalidExtension, Store().get_csv("test.txt"))
        self.assertRaises(InvalidExtension, Store().get_json("test.txt"))
        self.assertRaises(InvalidExtension, Store().get_pkl("test.txt"))

        self.assertRaises(FileNotFoundError, Store().get_csv("test.csv"))
        self.assertRaises(FileNotFoundError, Store().get_json("test.json"))
        self.assertRaises(FileNotFoundError, Store().get_pkl("test.pkl"))

    @raises(InvalidExtension, TypeError)
    def test_store_put_failures(self):
        self.assertRaises(InvalidExtension, Store().put_csv("test.txt", None))
        self.assertRaises(InvalidExtension, Store().put_json("test.txt", None))
        self.assertRaises(InvalidExtension, Store().put_pkl("test.txt", None))

        self.assertRaises(TypeError, Store().put_csv("test.csv", None))
        self.assertRaises(TypeError, Store().put_json("test.json", None))
        self.assertRaises(TypeError, Store().put_pkl("test.pkl", None))

    def test_get_and_put_dataframe(self):
        want = pd.DataFrame({"test": [1, 2, 3]})
        Store().put_csv("test.csv", want)
        got = Store().get_csv("test.csv")
        pd.testing.assert_frame_equal(got, want)

    def test_get_and_put_model(self):
        model = LogisticRegression()
        model.fit(
            [[0.1] for _ in range(500)] + [[0.9] for _ in range(500)],
            [1 for _ in range(500)] + [0 for _ in range(500)],
        )
        want = model.predict([[0.9], [0.9], [0.9]])

        Store().put_pkl("test.pkl", model)
        got = Store().get_pkl("test.pkl").predict([[0.9], [0.9], [0.9]])

        self.assertEqual(got.tolist(), want.tolist())

    def test_get_and_put_dict(self):
        want = {"auc": 0.9}
        Store().put_json("test.json", want)
        got = Store().get_json("test.json")
        self.assertEqual(got, want)
