from unittest import TestCase
from bloom.bloom import BloomFilter


class TestBloomFilter(TestCase):
    def test_basic_operators(self):
        filter = BloomFilter(n=100, fp_rate=0.01)
        assert "new element" not in filter

        filter.add("new element")
        assert "new element" in filter
