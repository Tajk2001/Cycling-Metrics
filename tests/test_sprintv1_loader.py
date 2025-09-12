import types
import pandas as pd
import numpy as np
import builtins
import importlib
import sys


def test_load_fit_to_dataframe_monkeypatch(monkeypatch):
    # Create a fake FitFile class and messages
    class FakeField:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class FakeMessage:
        def __init__(self, fields):
            self._fields = fields
        def __iter__(self):
            return iter(self._fields)

    class FakeFitFile:
        def __init__(self, path):
            self.path = path
        def get_messages(self, kind):
            if kind == 'record':
                # 5 seconds of data
                ts = pd.date_range("2025-01-01 06:00:00", periods=5, freq="1s")
                for t in ts:
                    yield FakeMessage([
                        FakeField('timestamp', t.to_pydatetime()),
                        FakeField('power', 200),
                        FakeField('cadence', 90),
                    ])
            elif kind == 'lap':
                st = pd.Timestamp('2025-01-01 06:00:01')
                et = pd.Timestamp('2025-01-01 06:00:04')
                yield FakeMessage([
                    FakeField('start_time', st.to_pydatetime()),
                    FakeField('timestamp', et.to_pydatetime()),
                ])
            else:
                return []

    # Monkeypatch SprintV1.FitFile
    import SprintV1
    monkeypatch.setattr(SprintV1, 'FitFile', FakeFitFile)

    df = SprintV1.load_fit_to_dataframe('fake.fit')
    assert isinstance(df, pd.DataFrame)
    assert 'power' in df.columns and 'cadence' in df.columns
    assert 'lap' in df.columns
    # Check lap assignment exists
    assert df['lap'].notna().any()

