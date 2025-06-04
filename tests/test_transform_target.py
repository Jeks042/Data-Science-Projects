import pandas as pd
import pandas.testing as pdt

from B1190869_Ezema_Chukwujekwu_ICA_Element_1 import transform_target


def test_transform_target_basic():
    df = pd.DataFrame({
        'ID': [1, 1, 2, 2, 3],
        'STATUS': ['X', '2', '0', 'C', '1']
    })
    result = transform_target(df)
    result = result.sort_values('ID').reset_index(drop=True)
    expected = pd.DataFrame({
        'ID': [1, 2, 3],
        'Target': [0, 0, 1]
    })
    pdt.assert_frame_equal(result, expected)

