"""
By definition, the only time you can be sure a Bloom filter is not working
correctly is when it gives a false positive. This test suite aims to stress
this be providing multiple lists of different data types in order to try to
force an error.
"""

from unittest import TestCase
from hypothesis import given
import hypothesis.strategies as st

from bloom.bloom import BloomFilter


FILTER_SIZE = 100
FP_RATE = 0.01


class TestFalseNegatives(TestCase):
    def setUp(self):
        self.filter = BloomFilter(n=FILTER_SIZE, fp_rate=FP_RATE)

    def tearDown(self):
        self.filter = None

    def test_basic_functionality(self):
        assert "new element" not in self.filter

        self.filter.add("new element")
        assert "new element" in self.filter

    @given(st.lists(st.integers(), min_size=FILTER_SIZE, unique=True))
    def test_integers(self, lst):
        for element in lst:
            self.filter.add(element)
            assert element in self.filter

        self.filter._clear()

    @given(
        st.lists(
            st.characters(min_codepoint=0, max_codepoint=256),
            min_size=FILTER_SIZE,
            unique=True,
        )
    )
    def test_ascii_characters(self, lst):
        for element in lst:
            self.filter.add(element)
            assert element in self.filter

        self.filter._clear()

    @given(st.lists(st.text(min_size=1), min_size=FILTER_SIZE, unique=True))
    def test_strings(self, lst):
        for element in lst:
            self.filter.add(element)
            assert element in self.filter

        self.filter._clear()
