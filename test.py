import unittest
import pandas as pd
import numpy as np
from detection_util import (
    start_condition,
    end_condition,
    detect_end,
    is_event,
    combine_close_events,
    )


class TestDetection(unittest.TestCase):
    def test_start_condition(self):
        df = pd.DataFrame(columns=['feces', 'urine'])
        df['feces'] = range(100)
        df['urine'] = range(100)
        self.assertTrue(start_condition(df, curr_idx=0, th_last=10))

        df = pd.DataFrame(columns=['feces', 'urine'])
        df['feces'] = list(range(5)) + [100]*5
        df['urine'] = list(range(5)) + [100]*5
        self.assertTrue(start_condition(df, curr_idx=0, th_last=5))
        self.assertFalse(start_condition(df, curr_idx=0, th_last=6))

        df = pd.DataFrame(columns=['feces', 'urine'])
        df['feces'] = [1]*100
        df['urine'] = [2]*100
        self.assertFalse(start_condition(df, curr_idx=0, th_last=10))

    def test_end_condition(self):
        df = pd.DataFrame(columns=['feces_deriv_2', 'urine_deriv_2'])
        df['feces_deriv_2'] = [-1]*100
        df['urine_deriv_2'] = [-1]*100
        self.assertTrue(end_condition(df, curr_idx=0, th_last=10))

        df = pd.DataFrame(columns=['feces_deriv_2', 'urine_deriv_2'])
        df['feces_deriv_2'] = [0] + [-1]*5 + [1]*5
        df['urine_deriv_2'] = [0] + [-1]*5 + [1]*5
        self.assertTrue(end_condition(df, curr_idx=0, th_last=5))
        self.assertFalse(end_condition(df, curr_idx=0, th_last=6))

        df = pd.DataFrame(columns=['feces_deriv_2', 'urine_deriv_2'])
        df['feces_deriv_2'] = [0] + [-1]*10
        df['urine_deriv_2'] = [0] + [-1]*5 + [1]*5
        self.assertFalse(end_condition(df, curr_idx=0, th_last=6))

    def test_detect_end(self):
        df = pd.DataFrame(columns=['feces_deriv_2', 'urine_deriv_2'])
        df['feces_deriv_2'] = [-1]*100
        df['urine_deriv_2'] = [-1]*100
        self.assertEqual(detect_end(df, start_idx=0, th_last=10), 1)

        df = pd.DataFrame(columns=['feces_deriv_2', 'urine_deriv_2'])
        df['feces_deriv_2'] = [0, 1, 2] + [-1]*5 + [1]*5
        df['urine_deriv_2'] = [0, 1, 2] + [-1]*5 + [1]*5
        self.assertEqual(detect_end(df, start_idx=0, th_last=5), 2)
        self.assertEqual(detect_end(df, start_idx=0, th_last=6), 12)

    def test_is_event(self):
        df = pd.DataFrame(columns=['feces', 'urine'])
        df['feces'] = np.arange(100)/100
        df['urine'] = np.arange(100)/100
        self.assertTrue(is_event(df, start_idx=0, end_idx=10))
        self.assertFalse(is_event(df, start_idx=0, end_idx=5))

        df = pd.DataFrame(columns=['feces', 'urine'])
        df['feces'] = np.arange(100)/1000
        df['urine'] = np.arange(100)/100
        self.assertTrue(is_event(df, start_idx=0, end_idx=10))

        df = pd.DataFrame(columns=['feces', 'urine'])
        df['feces'] = np.arange(100)/1000
        df['urine'] = np.arange(100)/1000
        self.assertFalse(is_event(df, start_idx=0, end_idx=10))

    def test_combine_close_events(self):
        start_idxes = [0, 5, 100, 280]
        end_idxes = [2, 8, 200, 300]
        res_s, res_e = combine_close_events(start_idxes, end_idxes)
        self.assertEqual(res_s, [0, 100, 280])
        self.assertEqual(res_e, [8, 200, 300])

        start_idxes = [0, 5, 68, 280]
        end_idxes = [2, 8, 200, 300]
        res_s, res_e = combine_close_events(start_idxes, end_idxes)
        self.assertEqual(res_s, [0, 68, 280])
        self.assertEqual(res_e, [8, 200, 300])

        start_idxes = [0, 5, 67, 280]
        end_idxes = [2, 8, 200, 300]
        res_s, res_e = combine_close_events(start_idxes, end_idxes)
        self.assertEqual(res_s, [0, 280])
        self.assertEqual(res_e, [200, 300])


if __name__ == '__main__':
    unittest.main()
